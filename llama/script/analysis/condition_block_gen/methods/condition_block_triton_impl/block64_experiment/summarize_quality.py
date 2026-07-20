"""Summarize paired LongBench quality runs for two block sizes.

The script deliberately treats generation timing as diagnostic only: the runs
keep condition-block statistics enabled so that their realized budgets can be
compared.  Use the decode-only latency harness for performance claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ....longbench.eval import read_jsonl, score_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args()


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _by_id(rows):
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample IDs in a quality result")
    return result


def summarize(dataset, baseline_rows, candidate_rows):
    baseline = _by_id(baseline_rows)
    candidate = _by_id(candidate_rows)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"Unpaired results: missing={missing}, extra={extra}")

    ordered_ids = list(baseline)
    exact = sum(
        baseline[sample_id]["pred"] == candidate[sample_id]["pred"]
        for sample_id in ordered_ids
    )
    baseline_ordered = [baseline[sample_id] for sample_id in ordered_ids]
    candidate_ordered = [candidate[sample_id] for sample_id in ordered_ids]
    baseline_score = score_rows(dataset, baseline_ordered, metadata={}, use_longbench_e=False)
    candidate_score = score_rows(dataset, candidate_ordered, metadata={}, use_longbench_e=False)
    return {
        "dataset": dataset,
        "samples": len(ordered_ids),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": round(candidate_score - baseline_score, 2),
        "exact_predictions": exact,
        "exact_fraction": exact / len(ordered_ids),
        "baseline_equiv_budget": _mean(baseline_ordered, "condition_block_equiv_budget"),
        "candidate_equiv_budget": _mean(candidate_ordered, "condition_block_equiv_budget"),
        "baseline_output_tokens": _mean(baseline_ordered, "output_tokens"),
        "candidate_output_tokens": _mean(candidate_ordered, "output_tokens"),
        "baseline_generation_seconds_diagnostic": _mean(
            baseline_ordered, "generation_seconds"
        ),
        "candidate_generation_seconds_diagnostic": _mean(
            candidate_ordered, "generation_seconds"
        ),
    }


def main():
    args = parse_args()
    summary = summarize(
        args.dataset,
        read_jsonl(args.baseline),
        read_jsonl(args.candidate),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
