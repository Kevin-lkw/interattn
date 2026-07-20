"""Check eager/CUDA-graph token parity for the best long-context preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from llama.script.analysis.condition_block_gen.common import load_model_and_tokenizer
from llama.script.analysis.condition_block_gen.longbench_v2_latency import (
    find_inputs_for_context,
    load_longbench_v2,
    make_generation_args,
)
from llama.script.analysis.condition_block_gen.methods import (
    build_method,
    generate_with_method,
)
from llama.script.analysis.runner_utils import set_seed

from .config import BEST_LONG_CONTEXT_CONFIG, configured_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _generation_namespace(args):
    config = BEST_LONG_CONTEXT_CONFIG
    return SimpleNamespace(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        full_attention_layers=0,
        condition_eps=config.eps,
        condition_block_size=config.block_size,
    )


def _generate(model, tokenizer, method, args, input_ids, attention_mask, *, cuda_graph):
    original_tokenizer_eos = tokenizer.eos_token_id
    original_model_eos = getattr(model.config, "eos_token_id", None)
    original_generation_eos = getattr(model.generation_config, "eos_token_id", None)
    try:
        # Force every path to execute exactly max_new_tokens; otherwise an EOS
        # can hide a later graph/eager mismatch.
        tokenizer.eos_token_id = None
        model.config.eos_token_id = None
        model.generation_config.eos_token_id = None
        with configured_environment(cuda_graph=cuda_graph, collect_stats=False):
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
        tokenizer.eos_token_id = original_tokenizer_eos
        model.config.eos_token_id = original_model_eos
        model.generation_config.eos_token_id = original_generation_eos


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("runtime_sanity requires a CUDA device")
    set_seed(args.seed)
    config = BEST_LONG_CONTEXT_CONFIG
    gen_args = _generation_namespace(args)
    load_args = make_generation_args(gen_args, config.method)
    model, tokenizer = load_model_and_tokenizer(load_args)
    method = build_method(load_args)
    repo, dataset = load_longbench_v2(args.hf_repo, args.split)
    max_position = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    rows = []
    for target in args.contexts:
        effective = int(target)
        if max_position:
            effective = min(effective, max_position - int(args.max_new_tokens))
        index, record, input_ids, attention_mask = find_inputs_for_context(
            dataset,
            tokenizer,
            effective,
            start_index=0,
            device=args.device,
        )
        eager_ids, eager_metadata = _generate(
            model,
            tokenizer,
            method,
            args,
            input_ids,
            attention_mask,
            cuda_graph=False,
        )
        graph_ids, graph_metadata = _generate(
            model,
            tokenizer,
            method,
            args,
            input_ids,
            attention_mask,
            cuda_graph=True,
        )
        row = {
            "target_context_tokens": int(target),
            "input_tokens": int(input_ids.shape[1]),
            "max_new_tokens": int(args.max_new_tokens),
            "record_index": int(index),
            "record_id": str(record.get("_id", index)),
            "eager_output_tokens": int(eager_ids.shape[1]),
            "graph_output_tokens": int(graph_ids.shape[1]),
            "tokens_exact": bool(torch.equal(eager_ids, graph_ids)),
            "eager_stats_disabled": bool(eager_metadata.get("condition_block_stats_disabled")),
            "graph_stats_disabled": bool(graph_metadata.get("condition_block_stats_disabled")),
            "graph_used": bool(graph_metadata.get("cuda_graph_decode")),
            "hf_repo": repo,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not row["tokens_exact"]:
            raise AssertionError(f"CUDA graph mismatch at context={target}: {row}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
