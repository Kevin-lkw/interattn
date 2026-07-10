import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...runner_utils import set_seed, str_to_torch_dtype
from ..common import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    _maybe_apply_chat_template,
    _middle_truncate_prompt,
    build_prompt,
    extract_answers,
    load_done_ids,
    read_records,
    record_id,
)
from ..methods.condition_ksim_cluster import generate_condition_ksim_cluster_cached
from .run import (
    DATASET2MAXLEN,
    LONGBENCH_DATASETS,
    load_longbench_records,
    longbench_prompt,
    resolve_dataset_name,
    resolve_model_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LongBench generation with condition-thresholded K-sim clusters."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--question-field", default="input")
    parser.add_argument("--answer-field", default="answers")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--id-field", default="_id")
    parser.add_argument("--full-attention-layers", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--condition-eps", type=float, default=0.1)
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=LONGBENCH_DATASETS,
    )
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--longbench-e", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    if args.cluster_size <= 0:
        parser.error("--cluster-size must be > 0")
    if args.kmeans_iters <= 0:
        parser.error("--kmeans-iters must be > 0")
    return args


def _dataset_name(record):
    dataset = str(record.get("dataset", "")).lower()
    if dataset.endswith("_e"):
        dataset = dataset[:-2]
    return dataset


def load_model_and_tokenizer(args):
    dtype = str_to_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "dtype": dtype,
        "device_map": {"": args.device},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is None:
        args.attn_implementation = "eager"
    elif args.attn_implementation != "eager":
        raise ValueError("condition_ksim_cluster requires --attn-implementation eager.")
    kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, tokenizer


def output_path(args, benchmark_name):
    model_name = str(args.model).rstrip("/").split("/")[-1]
    filename = (
        f"condition_ksim_cluster_size={args.cluster_size}"
        f"_iters={args.kmeans_iters}"
        f"_eps={args.condition_eps:g}.jsonl"
    )
    out_dir = args.output_root / model_name / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def pending_generation_records(args, benchmark_name, records=None):
    if records is None:
        if args.data is None:
            raise ValueError("--data is required unless LongBench records are loaded.")
        records = read_records(args.data, args.limit)
    elif args.limit is not None:
        records = records[: args.limit]
    out_path = output_path(args, benchmark_name)
    done_ids = load_done_ids(out_path)
    pending_records = [
        (index, record)
        for index, record in enumerate(records)
        if record_id(record, args, index) not in done_ids
    ]
    return records, out_path, done_ids, pending_records


def run_generation_benchmark(args, benchmark_name, records=None):
    set_seed(args.seed)
    records, out_path, done_ids, pending_records = pending_generation_records(
        args,
        benchmark_name,
        records,
    )
    if not pending_records:
        print(f"All predictions already exist; skipping model load: {out_path}")
        return out_path

    model, tokenizer = load_model_and_tokenizer(args)
    if args.max_input_tokens is None and getattr(model.config, "max_position_embeddings", None):
        args.max_input_tokens = int(model.config.max_position_embeddings)

    with out_path.open("a", encoding="utf-8") as handle:
        for index, record in tqdm(
            pending_records,
            desc=f"{benchmark_name}:condition_ksim_cluster",
            unit="sample",
            total=len(pending_records),
        ):
            rid = record_id(record, args, index)
            if rid in done_ids:
                continue
            prompt = build_prompt(record, args, longbench_prompt)
            prompt = _middle_truncate_prompt(prompt, tokenizer, args.max_input_tokens)
            prompt = _maybe_apply_chat_template(prompt, tokenizer, args, record)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            generation_start = time.perf_counter()
            output_ids, generation_metadata = generate_condition_ksim_cluster_cached(
                model=model,
                tokenizer=tokenizer,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get(
                    "attention_mask",
                    torch.ones_like(inputs["input_ids"], device=args.device),
                ),
                cluster_size=args.cluster_size,
                kmeans_iters=args.kmeans_iters,
                eps=args.condition_eps,
                delta_mode=args.delta_mode,
                max_new_tokens=args.max_new_tokens,
                full_attention_layers=args.full_attention_layers,
                device=args.device,
                dataset=_dataset_name(record),
            )
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            generation_seconds = time.perf_counter() - generation_start

            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            row = {
                "id": rid,
                "method": "condition_ksim_cluster",
                "pred": prediction,
                "answers": extract_answers(record, args),
                "all_classes": record.get("all_classes"),
                "length": record.get("length"),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(output_ids.shape[1]),
                "generation_seconds": generation_seconds,
                "cluster_size": int(args.cluster_size),
                "kmeans_iters": int(args.kmeans_iters),
                "condition_eps": float(args.condition_eps),
                "delta_mode": str(args.delta_mode),
            }
            row.update(generation_metadata)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done_ids.add(rid)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"Saved predictions to: {out_path}")
    return out_path


def main():
    args = resolve_model_config(parse_args())
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET2MAXLEN[args.dataset]
    records = None if args.data is not None else load_longbench_records(args)
    benchmark_name = f"longbench/{resolve_dataset_name(args.dataset, args.longbench_e)}"
    run_generation_benchmark(args, benchmark_name=benchmark_name, records=records)


if __name__ == "__main__":
    main()
