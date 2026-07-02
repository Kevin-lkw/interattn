import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize condition_block_flex_stage_latency.py JSONL output."
    )
    parser.add_argument("jsonl", type=Path)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def main():
    args = parse_args()
    rows = []
    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != "sanity":
                rows.append(row)

    full_by_context = {}
    dense_by_context = {}
    by_key = defaultdict(dict)
    for row in rows:
        context = int(row["context_tokens"])
        stage = row["stage"]
        if stage == "full_sdpa_decode_attention":
            full_by_context[context] = float(row["latency_ms"])
            continue
        if stage == "flex_dense_decode_attention":
            dense_by_context[context] = row
            continue
        ratio = float(row.get("selected_ratio", 0.0))
        by_key[(context, ratio)][stage] = row

    print(
        "context selected full_ms flex_dense_ms triton_ms flex_compact_ms "
        "flex_blockmask_ms flex_from_kv_ms compact_build_ms compact_tokens"
    )
    for (context, ratio), stages in sorted(by_key.items()):
        triton = stages.get("triton_dummy_hybrid_attention", {})
        compact = stages.get("flex_compact_hybrid_attention", {})
        block = stages.get("flex_block_sparse_token_attention", {})
        from_kv = stages.get("flex_block_sparse_from_kv_blocks_attention", {})
        dense = dense_by_context.get(context, {})
        print(
            f"{context} "
            f"{ratio:g} "
            f"{fmt(full_by_context.get(context))} "
            f"{fmt(dense.get('latency_ms'))} "
            f"{fmt(triton.get('latency_ms'))} "
            f"{fmt(compact.get('latency_ms'))} "
            f"{fmt(block.get('latency_ms'))} "
            f"{fmt(from_kv.get('latency_ms'))} "
            f"{fmt(compact.get('compact_build_ms'))} "
            f"{compact.get('compact_kv_tokens', '-')}"
        )


if __name__ == "__main__":
    main()
