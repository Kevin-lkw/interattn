import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize condition-block IO-bound analysis JSONL outputs into "
            "Markdown tables for README updates."
        )
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--dummy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_seconds(value):
    return f"{float(value):.3f}s"


def fmt_ms(value):
    return f"{float(value):.3f} ms"


def fmt_speedup(value):
    return f"{float(value):.2f}x"


def measured(rows):
    return [row for row in rows if row.get("phase") == "measured"]


def summarize_decode(rows):
    by_context = defaultdict(dict)
    for row in measured(rows):
        by_context[int(row["target_context_tokens"])][row["method"]] = row

    lines = [
        "### Decode-only full vs Triton",
        "",
        "| context | full decode | Triton decode | Triton vs full |",
        "|---:|---:|---:|---:|",
    ]
    for context in sorted(by_context):
        full = by_context[context].get("full")
        triton = by_context[context].get("condition_block_triton")
        if not full or not triton:
            continue
        full_decode = float(full["decode_only_seconds"])
        triton_decode = float(triton["decode_only_seconds"])
        lines.append(
            f"| {context // 1024}K | {fmt_seconds(full_decode)} | "
            f"{fmt_seconds(triton_decode)} | {fmt_speedup(full_decode / triton_decode)} |"
        )
    return "\n".join(lines)


def summarize_stage(rows):
    by_context = defaultdict(dict)
    sparse_rows = defaultdict(dict)
    for row in rows:
        context = int(row["context_tokens"])
        stage = row["stage"]
        if stage == "dummy_sparse_attention":
            sparse_rows[context][float(row["selected_ratio"])] = row
        else:
            by_context[context][stage] = row

    lines = [
        "### Stage microbench",
        "",
        "| context | selection materialize | full SDPA attn | dummy sparse 0% | dummy sparse 10% | dummy sparse 25% |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for context in sorted(set(by_context) | set(sparse_rows)):
        stages = by_context[context]
        sparse = sparse_rows[context]
        lines.append(
            f"| {context // 1024}K | "
            f"{fmt_ms(stages['selection_materialize']['latency_ms'])} | "
            f"{fmt_ms(stages['full_sdpa_decode_attention']['latency_ms'])} | "
            f"{fmt_ms(sparse[0.0]['latency_ms'])} | "
            f"{fmt_ms(sparse[0.1]['latency_ms'])} | "
            f"{fmt_ms(sparse[0.25]['latency_ms'])} |"
        )
    return "\n".join(lines)


def summarize_dummy(rows):
    by_context = defaultdict(dict)
    for row in measured(rows):
        by_context[int(row["target_context_tokens"])][row["config"]] = row

    lines = [
        "### Dummy-selection decode-only",
        "",
        "| context | production | dummy 0% | dummy 10% | dummy 25% |",
        "|---:|---:|---:|---:|---:|",
    ]
    for context in sorted(by_context):
        configs = by_context[context]
        production = float(configs["production"]["decode_only_seconds"])
        dummy0 = float(configs["dummy_ratio_0"]["decode_only_seconds"])
        dummy10 = float(configs["dummy_ratio_0.1"]["decode_only_seconds"])
        dummy25 = float(configs["dummy_ratio_0.25"]["decode_only_seconds"])
        lines.append(
            f"| {context // 1024}K | {fmt_seconds(production)} | "
            f"{fmt_seconds(dummy0)} ({fmt_speedup(production / dummy0)}) | "
            f"{fmt_seconds(dummy10)} ({fmt_speedup(production / dummy10)}) | "
            f"{fmt_seconds(dummy25)} ({fmt_speedup(production / dummy25)}) |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    sections = [
        summarize_stage(read_jsonl(args.stage)),
        summarize_decode(read_jsonl(args.decode)),
        summarize_dummy(read_jsonl(args.dummy)),
    ]
    text = "\n\n".join(sections) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
