#!/usr/bin/env python3
"""Estimate and measure Llama-3.1-8B attention CUDA memory.

The default is a 32K-token prefill using PyTorch SDPA forced onto its fused
Flash Attention kernel. Pass --load-model to also measure the HF checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Llama31Config:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 128256
    head_dim: int = 128
    tie_word_embeddings: bool = False


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def gib(nbytes: int | float) -> float:
    return float(nbytes) / 1024**3


def mib(nbytes: int | float) -> float:
    return float(nbytes) / 1024**2


def parameter_count(c: Llama31Config) -> int:
    # Llama linear layers have no bias. Each decoder layer has two RMSNorms.
    kv_width = c.num_key_value_heads * c.head_dim
    attention = (
        c.hidden_size * (c.num_attention_heads * c.head_dim)
        + 2 * c.hidden_size * kv_width
        + c.hidden_size * (c.num_attention_heads * c.head_dim)
    )
    mlp = 3 * c.hidden_size * c.intermediate_size
    decoder = c.num_hidden_layers * (attention + mlp + 2 * c.hidden_size)
    embeddings = c.vocab_size * c.hidden_size
    lm_head = 0 if c.tie_word_embeddings else embeddings
    return embeddings + decoder + c.hidden_size + lm_head


def print_estimates(
    c: Llama31Config, batch: int, seq: int, dtype: torch.dtype, implementation: str
) -> None:
    e = torch.empty((), dtype=dtype).element_size()
    q = batch * c.num_attention_heads * seq * c.head_dim * e
    k = batch * c.num_key_value_heads * seq * c.head_dim * e
    v = k
    matrix = batch * c.num_attention_heads * seq * seq * e
    fp32_matrix = batch * c.num_attention_heads * seq * seq * 4
    kv_cache = (k + v) * c.num_hidden_layers
    params = parameter_count(c)

    print("\n=== Analytical sizes (dense tensors, GiB = 2^30 bytes) ===")
    print(f"dtype={dtype}, batch={batch}, sequence={seq}")
    print(f"Model parameters : {params:,} params = {gib(params * e):8.3f} GiB")
    print(f"Q (one layer)    : {mib(q):8.2f} MiB")
    print(f"K (one layer)    : {mib(k):8.2f} MiB")
    print(f"V (one layer)    : {mib(v):8.2f} MiB")
    print(f"Q+K+V            : {mib(q + k + v):8.2f} MiB")
    allocation = "allocated by eager" if implementation == "eager" else "conceptual; not materialized by Flash"
    print(f"Attention matrix : {gib(matrix):8.3f} GiB  ({allocation})")
    if implementation == "eager":
        print(f"FP32 matrix      : {gib(fp32_matrix):8.3f} GiB  (softmax temporary)")
    print(f"KV cache         : {gib(kv_cache):8.3f} GiB  (all {c.num_hidden_layers} layers)")
    print("Note: layers execute sequentially, so QKV/attention matrices are not multiplied by 32.")
    print("      KV cache is multiplied by 32 because it persists during generation.")


def cleanup(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def allocated(device: torch.device) -> int:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device)


def contained_tensor_bytes(obj: object) -> int:
    """Count distinct tensor payloads contained in a HF cache object."""
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(value: object) -> int:
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, torch.Tensor):
            # A cache should not contain views, but data_ptr avoids double
            # counting if two attributes happen to reference one tensor.
            pointer = value.data_ptr()
            if pointer in seen_tensors:
                return 0
            seen_tensors.add(pointer)
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        if hasattr(value, "__dict__"):
            return visit(vars(value))
        return 0

    return visit(obj)


@torch.inference_mode()
def microbenchmark(c: Llama31Config, batch: int, seq: int, dtype: torch.dtype, device: torch.device) -> None:
    """Run the same dense operations as HF eager attention for one layer."""
    cleanup(device)
    baseline = allocated(device)
    try:
        q = torch.empty(
            (batch, c.num_attention_heads, seq, c.head_dim), dtype=dtype, device=device
        ).normal_()
        k = torch.empty(
            (batch, c.num_key_value_heads, seq, c.head_dim), dtype=dtype, device=device
        ).normal_()
        v = torch.empty_like(k).normal_()
        after_qkv = allocated(device)

        groups = c.num_attention_heads // c.num_key_value_heads
        k_repeated = k.repeat_interleave(groups, dim=1)
        v_repeated = v.repeat_interleave(groups, dim=1)
        # A bool mask lets us model causal attention without another full
        # floating-point [batch, heads, seq, seq] allocation.
        causal_mask = torch.ones((seq, seq), dtype=torch.bool, device=device).triu_(1)
        before_attention = allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        scores = torch.matmul(q, k_repeated.transpose(-2, -1)) / math.sqrt(c.head_dim)
        scores.masked_fill_(causal_mask, float("-inf"))
        # Hugging Face Llama eager attention computes softmax in FP32 and then
        # converts probabilities back to the query dtype.
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        output = torch.matmul(probabilities, v_repeated)
        del output
        peak = torch.cuda.max_memory_allocated(device)

        print("\n=== CUDA eager-attention microbenchmark (one layer) ===")
        print(f"Q+K+V allocated             : {mib(after_qkv - baseline):8.2f} MiB")
        print(f"Repeated K/V + causal mask  : {mib(before_attention - after_qkv):8.2f} MiB")
        print(f"Attention workspace peak    : {gib(peak - before_attention):8.3f} GiB")
        print(f"Total peak above baseline   : {gib(peak - baseline):8.3f} GiB")
    except torch.OutOfMemoryError as exc:
        print("\nCUDA OOM during the one-layer eager microbenchmark.")
        print(f"Allocator message: {exc}")
    finally:
        # Locals may not all exist if allocation failed.
        for name in ("q", "k", "v", "k_repeated", "v_repeated", "causal_mask", "scores", "probabilities"):
            if name in locals():
                del locals()[name]
        cleanup(device)


@torch.inference_mode()
def flash_microbenchmark(
    c: Llama31Config, batch: int, seq: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Force PyTorch SDPA to use its fused Flash Attention CUDA backend."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cleanup(device)
    baseline = allocated(device)
    try:
        q = torch.empty(
            (batch, c.num_attention_heads, seq, c.head_dim), dtype=dtype, device=device
        ).normal_()
        k = torch.empty(
            (batch, c.num_key_value_heads, seq, c.head_dim), dtype=dtype, device=device
        ).normal_()
        v = torch.empty_like(k).normal_()
        after_qkv = allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        # enable_gqa=True keeps the 8 KV heads; no explicit 8 -> 32 head copy.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
            )
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)

        print("\n=== CUDA fused Flash SDPA microbenchmark (one layer) ===")
        print(f"Q+K+V allocated             : {mib(after_qkv - baseline):8.2f} MiB")
        print(f"Flash workspace + output peak: {mib(peak - after_qkv):7.2f} MiB")
        print(f"Total peak above baseline   : {gib(peak - baseline):8.3f} GiB")
        del output, q, k, v
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        print("\nFused Flash SDPA could not run; no fallback was allowed.")
        print(f"Backend message: {exc}")
    finally:
        cleanup(device)


@torch.inference_mode()
def full_model_benchmark(
    model_name: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    implementation: str,
    use_cache: bool,
) -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("--load-model requires: pip install transformers accelerate") from exc

    cleanup(device)
    before_load = allocated(device)
    print(f"\nLoading {model_name!r} with attn_implementation={implementation!r} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    after_load = allocated(device)
    actual_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    input_ids = torch.zeros((batch, seq), dtype=torch.long, device=device)
    before_forward = allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Run the base model to avoid allocating [batch, seq, vocab] LM logits.
    base_model = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    if implementation == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        attention_context = nullcontext()
    with attention_context:
        outputs = base_model(input_ids=input_ids, use_cache=use_cache, return_dict=True)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    cache = getattr(outputs, "past_key_values", None)
    actual_cache_bytes = contained_tensor_bytes(cache) if cache is not None else 0

    print("\n=== Full model CUDA measurement ===")
    print(f"Parameter tensors (exact)    : {gib(actual_param_bytes):8.3f} GiB")
    print(f"CUDA allocated by model load : {gib(after_load - before_load):8.3f} GiB")
    print(f"CUDA allocated pre-forward   : {gib(before_forward):8.3f} GiB")
    print(f"Forward incremental peak     : {gib(peak - before_forward):8.3f} GiB")
    print(f"Total torch allocated peak   : {gib(peak):8.3f} GiB")
    print(f"Returned KV cache (exact)    : {gib(actual_cache_bytes):8.3f} GiB")
    print("The forward peak includes QKV, attention temporaries, hidden states, and framework buffers.")
    del outputs, input_ids, base_model, model
    cleanup(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
        help="Default sdpa is forced onto PyTorch's fused Flash CUDA backend.",
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="Skip all CUDA allocations."
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Also download/load the checkpoint and measure a full prefill.",
    )
    parser.add_argument(
        "--no-kv-cache",
        dest="use_cache",
        action="store_false",
        help="Disable the KV cache during the full-model prefill (cache is enabled by default).",
    )
    parser.set_defaults(use_cache=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0 or args.batch_size <= 0:
        raise SystemExit("--seq-len and --batch-size must be positive")
    dtype = DTYPES[args.dtype]
    config = Llama31Config()
    print_estimates(config, args.batch_size, args.seq_len, dtype, args.attn_implementation)

    if args.estimate_only:
        return
    if not torch.cuda.is_available():
        print("\nCUDA is unavailable; analytical estimates were printed, measurements skipped.")
        return
    device = torch.device(args.device)
    if args.attn_implementation == "eager":
        microbenchmark(config, args.batch_size, args.seq_len, dtype, device)
    elif args.attn_implementation == "sdpa":
        flash_microbenchmark(config, args.batch_size, args.seq_len, dtype, device)
    else:
        print("\nThe isolated benchmark uses PyTorch SDPA; skipping it for flash_attention_2.")
    if args.load_model:
        full_model_benchmark(
            args.model,
            args.batch_size,
            args.seq_len,
            dtype,
            device,
            args.attn_implementation,
            args.use_cache,
        )


if __name__ == "__main__":
    main()
