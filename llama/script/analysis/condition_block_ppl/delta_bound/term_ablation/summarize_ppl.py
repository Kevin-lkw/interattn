"""Compare term-ablation PPL sweeps at matched measured attention budgets."""

import argparse
import bisect
import json
import math
from pathlib import Path

import torch


T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}
CONFIG_KEYS = (
    "model",
    "dataset",
    "dtype",
    "seq_len",
    "num_samples",
    "start_offset",
    "sample_stride",
    "ppl_protocol",
    "block_size",
    "delta_mode",
    "full_attention_layers",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=None)
    parser.add_argument("--num-budgets", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_summaries(items):
    summaries = {}
    for label, raw_path in items:
        if label in summaries:
            raise ValueError(f"Duplicate summary label: {label}")
        path = Path(raw_path)
        summaries[label] = torch.load(path, map_location="cpu", weights_only=False)
    return summaries


def _validate(summaries, baseline):
    if baseline not in summaries:
        raise ValueError(f"Unknown baseline label: {baseline}")
    reference = summaries[baseline]
    for label, summary in summaries.items():
        if summary.get("starts") != reference.get("starts"):
            raise ValueError(f"Sample starts differ for {label}")
        for key in CONFIG_KEYS:
            if summary.get("config", {}).get(key) != reference.get("config", {}).get(key):
                raise ValueError(f"Config field {key!r} differs for {label}")
        if set(summary.get("samples", {})) != set(reference.get("samples", {})):
            raise ValueError(f"Completed samples differ for {label}")


def _curve(record):
    by_budget = {}
    for result in record["results"].values():
        budget = float(result["measured_budget"])
        by_budget.setdefault(budget, []).append(float(result["student_nll"]))
    return sorted(
        (budget, sum(values) / len(values))
        for budget, values in by_budget.items()
    )


def interpolate_curve(points, budget):
    """Linearly interpolate NLL on a monotone measured-budget curve."""
    if not points:
        raise ValueError("Cannot interpolate an empty curve")
    xs = [point[0] for point in points]
    if budget < xs[0] - 1e-12 or budget > xs[-1] + 1e-12:
        raise ValueError(f"Budget {budget} is outside [{xs[0]}, {xs[-1]}]")
    idx = bisect.bisect_left(xs, budget)
    if idx == 0:
        return points[0][1]
    if idx == len(points):
        return points[-1][1]
    x0, y0 = points[idx - 1]
    x1, y1 = points[idx]
    if math.isclose(x0, x1):
        return 0.5 * (y0 + y1)
    alpha = (budget - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)


def _mean_interval(values):
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean]}
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    critical = T95.get(n, 1.96)
    half = critical * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def _common_range(summaries):
    lows = []
    highs = []
    for summary in summaries.values():
        for sample in summary["samples"].values():
            points = _curve(sample)
            lows.append(points[0][0])
            highs.append(points[-1][0])
    low = max(lows)
    high = min(highs)
    if low > high:
        raise ValueError(f"PPL curves have no common budget range: [{low}, {high}]")
    return low, high


def _default_budgets(low, high, count):
    if count <= 0:
        raise ValueError("--num-budgets must be > 0")
    if count == 1 or math.isclose(low, high):
        return [0.5 * (low + high)]
    return [low + idx * (high - low) / (count - 1) for idx in range(count)]


def compare(summaries, baseline, budgets=None, num_budgets=5):
    _validate(summaries, baseline)
    low, high = _common_range(summaries)
    if budgets is None:
        budgets = _default_budgets(low, high, num_budgets)
    for budget in budgets:
        if budget < low - 1e-12 or budget > high + 1e-12:
            raise ValueError(f"Requested budget {budget} is outside [{low}, {high}]")

    reference = summaries[baseline]
    output = {
        "baseline": baseline,
        "common_budget_range": [low, high],
        "budgets": [float(value) for value in budgets],
        "config": {key: reference["config"].get(key) for key in CONFIG_KEYS},
        "results": {},
    }
    for budget in budgets:
        sample_nlls = {label: [] for label in summaries}
        teacher_nlls = []
        token_counts = []
        for sample_key, baseline_sample in reference["samples"].items():
            teacher_nlls.append(float(baseline_sample["teacher_nll"]))
            token_counts.append(int(baseline_sample.get("num_tokens", 1)))
            for label, summary in summaries.items():
                sample = summary["samples"][sample_key]
                sample_nlls[label].append(interpolate_curve(_curve(sample), budget))

        total_tokens = sum(token_counts)
        teacher_corpus_nll = sum(
            nll * count for nll, count in zip(teacher_nlls, token_counts)
        ) / total_tokens
        baseline_values = sample_nlls[baseline]
        rows = {}
        for label, values in sample_nlls.items():
            corpus_nll = sum(
                nll * count for nll, count in zip(values, token_counts)
            ) / total_tokens
            deltas = [
                value - base for value, base in zip(values, baseline_values)
            ]
            excess_base = sum(
                (base - teacher) * count
                for base, teacher, count in zip(
                    baseline_values, teacher_nlls, token_counts
                )
            ) / total_tokens
            excess_value = corpus_nll - teacher_corpus_nll
            rows[label] = {
                "corpus_nll": corpus_nll,
                "corpus_ppl": math.exp(corpus_nll),
                "ppl_ratio_baseline": math.exp(
                    corpus_nll
                    - sum(
                        base * count
                        for base, count in zip(baseline_values, token_counts)
                    )
                    / total_tokens
                ),
                "paired_delta_nll": _mean_interval(deltas),
                "excess_nll_ratio_baseline": (
                    excess_value / excess_base
                    if not math.isclose(excess_base, 0.0, abs_tol=1e-15)
                    else None
                ),
            }
        output["results"][f"{budget:.12g}"] = {
            "budget": float(budget),
            "teacher_corpus_ppl": math.exp(teacher_corpus_nll),
            "methods": rows,
        }
    return output


def _markdown(summary):
    config = summary["config"]
    low, high = summary["common_budget_range"]
    labels = list(next(iter(summary["results"].values()))["methods"])
    lines = [
        "# Matched-budget PPL comparison",
        "",
        f"Model `{config['model']}`, aligned windows `{config['num_samples']} x "
        f"{config['seq_len']}` tokens, block size `{config['block_size']}`. "
        f"Common measured-budget range: `{low:.6f}` to `{high:.6f}`.",
        "",
        "NLL is linearly interpolated within each window before aggregation. "
        "Intervals are paired 95% Student-t intervals over windows.",
        "",
        "| budget | method | corpus PPL | PPL/base | excess NLL/base | delta NLL 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for result in summary["results"].values():
        for label in labels:
            row = result["methods"][label]
            low_ci, high_ci = row["paired_delta_nll"]["ci95"]
            excess = row["excess_nll_ratio_baseline"]
            excess_text = "n/a" if excess is None else f"{excess:.3f}"
            lines.append(
                f"| {result['budget']:.6f} | {label} | {row['corpus_ppl']:.5f} | "
                f"{row['ppl_ratio_baseline']:.5f} | {excess_text} | "
                f"[{low_ci:+.5f}, {high_ci:+.5f}] |"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summaries = _load_summaries(args.summary)
    result = compare(
        summaries,
        baseline=args.baseline,
        budgets=args.budgets,
        num_budgets=args.num_budgets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    markdown_path.write_text(_markdown(result))
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
