import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..runner_utils import set_seed, str_to_torch_dtype
from .common import DEFAULT_MODEL


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "result" / "generate" / "prefill_decode_profile.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure long-context prefill latency separately from cached single-token "
            "decode latency for autoregressive generation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[16 * 1024])
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional text prompt source. Tokens are repeated or truncated to the "
            "requested --prompt-tokens lengths. Without this, synthetic token ids are used."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--print-per-repeat",
        action="store_true",
        help="Print each repeat row in addition to aggregate rows.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "dtype": str_to_torch_dtype(args.dtype),
        "device_map": {"": args.device},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is not None:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, tokenizer


def build_prompt_ids(tokenizer, prompt_tokens, prompt_file, seed):
    if prompt_file is not None:
        text = prompt_file.read_text(encoding="utf-8")
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.numel() == 0:
            raise ValueError(f"{prompt_file} did not produce any tokens.")
        repeats = (prompt_tokens + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeats)[:prompt_tokens]
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + prompt_tokens)
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
        ids = torch.randint(0, vocab_size, (prompt_tokens,), generator=generator, dtype=torch.long)
        if tokenizer.bos_token_id is not None:
            ids[0] = int(tokenizer.bos_token_id)

    return ids.unsqueeze(0)


def synchronize_if_needed(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_call(device, fn):
    if str(device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device)
        return result, start.elapsed_time(end) / 1000.0

    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


@torch.inference_mode()
def run_once(model, input_ids, device, decode_steps):
    attention_mask = torch.ones_like(input_ids, device=device)

    def prefill():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    prefill_outputs, prefill_seconds = timed_call(device, prefill)
    past_key_values = prefill_outputs.past_key_values
    next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    del prefill_outputs

    def decode():
        nonlocal past_key_values, attention_mask, next_token
        for _ in range(decode_steps):
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=device)],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        return next_token

    _, decode_seconds = timed_call(device, decode)
    total_seconds = prefill_seconds + decode_seconds
    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_profiled_seconds": total_seconds,
        "decode_seconds_per_token": decode_seconds / max(decode_steps, 1),
        "prefill_pct": 100.0 * prefill_seconds / total_seconds if total_seconds else 0.0,
        "decode_pct": 100.0 * decode_seconds / total_seconds if total_seconds else 0.0,
    }


def summarize(rows):
    keys = [
        "prefill_seconds",
        "decode_seconds",
        "total_profiled_seconds",
        "decode_seconds_per_token",
        "prefill_pct",
        "decode_pct",
    ]
    summary = {}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        summary[f"{key}_mean"] = float(values.mean().item())
        summary[f"{key}_std"] = float(values.std(unbiased=False).item())
    return summary


def format_summary(row):
    return (
        f"prompt={row['prompt_tokens']:>6} decode_steps={row['decode_steps']:>4} "
        f"prefill={row['prefill_seconds_mean']:.3f}s "
        f"decode={row['decode_seconds_mean']:.3f}s "
        f"decode/token={1000.0 * row['decode_seconds_per_token_mean']:.2f}ms "
        f"decode_pct={row['decode_pct_mean']:.1f}%"
    )


def validate_args(args):
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if any(value <= 0 for value in args.prompt_tokens):
        raise ValueError("--prompt-tokens values must be > 0")
    if any(value < 0 for value in args.decode_steps):
        raise ValueError("--decode-steps values must be >= 0")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for prompt_tokens in args.prompt_tokens:
            input_ids = build_prompt_ids(tokenizer, prompt_tokens, args.prompt_file, args.seed).to(device)

            for decode_steps in args.decode_steps:
                for _ in range(args.warmup):
                    _ = run_once(model, input_ids, device, decode_steps)
                    synchronize_if_needed(device)
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()

                repeat_rows = []
                for repeat in range(args.repeats):
                    if str(device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    row = run_once(model, input_ids, device, decode_steps)
                    row.update(
                        {
                            "type": "repeat",
                            "model": args.model,
                            "dtype": args.dtype,
                            "attn_implementation": args.attn_implementation,
                            "prompt_tokens": prompt_tokens,
                            "decode_steps": decode_steps,
                            "repeat": repeat,
                        }
                    )
                    if str(device).startswith("cuda"):
                        row["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    repeat_rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if args.print_per_repeat:
                        print(
                            f"repeat={repeat} prompt={prompt_tokens} decode_steps={decode_steps} "
                            f"prefill={row['prefill_seconds']:.3f}s "
                            f"decode={row['decode_seconds']:.3f}s "
                            f"decode_pct={row['decode_pct']:.1f}%"
                        )

                aggregate = {
                    "type": "aggregate",
                    "model": args.model,
                    "dtype": args.dtype,
                    "attn_implementation": args.attn_implementation,
                    "prompt_tokens": prompt_tokens,
                    "decode_steps": decode_steps,
                    "repeats": args.repeats,
                    **summarize(repeat_rows),
                }
                if any("peak_memory_gb" in row for row in repeat_rows):
                    aggregate["peak_memory_gb_max"] = max(row.get("peak_memory_gb", 0.0) for row in repeat_rows)
                handle.write(json.dumps(aggregate) + "\n")
                handle.flush()
                print(format_summary(aggregate))

    print(f"Saved profile rows to: {args.output}")


if __name__ == "__main__":
    main()
