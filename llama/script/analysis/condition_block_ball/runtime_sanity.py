"""Eager vs CUDA-graph token parity for the diag_ell fused production path.

Monkeypatches `core._run_condition_block_selection_stats` with the diag_ell
stats runner, then generates a fixed number of tokens twice (eager decode and
CUDA-graph decode) on real LongBench-v2 prompts and asserts the token sequences
match exactly. Mirrors `best_long_context/runtime_sanity.py`, but for the
default block-32 FP32-summary config with diag_ell selection.
"""

import argparse
import contextlib
import json
import os
from types import SimpleNamespace

import torch

from ..condition_block_gen.common import load_model_and_tokenizer
from ..condition_block_gen.longbench_v2_latency import (
    find_inputs_for_context,
    load_longbench_v2,
    make_generation_args,
)
from ..condition_block_gen.methods import build_method, generate_with_method
from ..condition_block_gen.methods.condition_block_triton_impl import core
from ..runner_utils import set_seed
from .triton_selection import run_selection_stats_diag_ell


@contextlib.contextmanager
def _decode_env(cuda_graph):
    keys = {
        "CONDITION_BLOCK_SKIP_STATS": "1",
        "CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE": "1",
        "CONDITION_BLOCK_CUDA_GRAPH": "1" if cuda_graph else "0",
    }
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v2", action="store_true", help="Use the persistent v2 stats kernel.")
    parser.add_argument("--v3", action="store_true", help="Use the tensor-core v3 stats kernel.")
    parser.add_argument(
        "--finv2", action="store_true", help="Also use the persistent finalize (implies --v3)."
    )
    return parser.parse_args()


def _generate(model, tokenizer, method, args, input_ids, attention_mask, *, cuda_graph):
    saved_eos = (
        tokenizer.eos_token_id,
        getattr(model.config, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    )
    try:
        tokenizer.eos_token_id = None
        model.config.eos_token_id = None
        model.generation_config.eos_token_id = None
        with _decode_env(cuda_graph):
            output_ids, metadata = generate_with_method(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                method=method,
                device=args.device,
                dataset="longbench_v2",
            )
        torch.cuda.synchronize(input_ids.device)
        return output_ids.detach().cpu(), metadata
    finally:
        tokenizer.eos_token_id = saved_eos[0]
        model.config.eos_token_id = saved_eos[1]
        model.generation_config.eos_token_id = saved_eos[2]


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.finv2:
        from .triton_finalize_v2 import decode_output_fused_v2
        from .triton_selection_v3 import run_selection_stats_diag_ell_v3

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v3
        core._condition_block_decode_output_fused_triton = decode_output_fused_v2
    elif args.v3:
        from .triton_selection_v3 import run_selection_stats_diag_ell_v3

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v3
    elif args.v2:
        from .triton_selection_v2 import run_selection_stats_diag_ell_v2

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v2
    else:
        core._run_condition_block_selection_stats = run_selection_stats_diag_ell
    gen_args = SimpleNamespace(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        full_attention_layers=0,
        condition_eps=args.eps,
        condition_block_size=args.block_size,
    )
    load_args = make_generation_args(gen_args, "condition_block_triton")
    model, tokenizer = load_model_and_tokenizer(load_args)
    method = build_method(load_args)
    _repo, dataset = load_longbench_v2(None, args.split)
    max_position = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    for target in args.contexts:
        effective = int(target)
        if max_position:
            effective = min(effective, max_position - int(args.max_new_tokens))
        index, record, input_ids, attention_mask = find_inputs_for_context(
            dataset, tokenizer, effective, start_index=0, device=args.device
        )
        eager_ids, _ = _generate(
            model, tokenizer, method, args, input_ids, attention_mask, cuda_graph=False
        )
        graph_ids, graph_meta = _generate(
            model, tokenizer, method, args, input_ids, attention_mask, cuda_graph=True
        )
        row = {
            "context": int(target),
            "input_tokens": int(input_ids.shape[1]),
            "tokens_exact": bool(torch.equal(eager_ids, graph_ids)),
            "graph_used": bool(graph_meta.get("cuda_graph_decode")),
        }
        print(json.dumps(row), flush=True)
        assert row["tokens_exact"], f"diag_ell eager/graph token mismatch at {target}"
        assert row["graph_used"], f"CUDA graph not active at {target}"
    print("DIAG_ELL RUNTIME SANITY PASSED")


if __name__ == "__main__":
    main()
