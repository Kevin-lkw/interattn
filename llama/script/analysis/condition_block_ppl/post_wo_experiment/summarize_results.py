"""Create a compact matched-budget report from completed experiment outputs."""

import argparse
import json
import math
from pathlib import Path

import torch

from .core import CONDITION_VARIANTS


def parse_args():
    base = Path(__file__).resolve().parents[4] / "result" / "post_wo_condition_block"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppl-summary", type=Path, default=base / "ppl_n5" / "summary.pt"
    )
    parser.add_argument(
        "--spearman-summary",
        type=Path,
        default=base / "offline_spearman" / "summary.json",
    )
    parser.add_argument("--output", type=Path, default=base / "report.json")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.23, 0.25, 0.28, 0.30, 0.33, 0.37, 0.40, 0.45],
    )
    return parser.parse_args()


def _interpolate_nll(settings, budget):
    points = sorted(
        (
            float(record["mean_measured_budget"]),
            float(record["corpus_nll"]),
        )
        for record in settings.values()
    )
    if budget < points[0][0] or budget > points[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= budget <= x1:
            alpha = 0.0 if x1 == x0 else (budget - x0) / (x1 - x0)
            nll = y0 + alpha * (y1 - y0)
            return {"nll": nll, "ppl": math.exp(nll)}
    if budget == points[-1][0]:
        return {"nll": points[-1][1], "ppl": math.exp(points[-1][1])}
    return None


def main():
    args = parse_args()
    ppl = torch.load(args.ppl_summary, map_location="cpu", weights_only=False)
    spearman = json.loads(args.spearman_summary.read_text())
    matched = {}
    for budget in args.budgets:
        row = {}
        for variant in CONDITION_VARIANTS:
            interpolated = _interpolate_nll(
                ppl["aggregate"]["variants"][variant], float(budget)
            )
            if interpolated is not None:
                row[variant] = interpolated
        if len(row) == len(CONDITION_VARIANTS):
            matched[str(float(budget))] = row

    report = {
        "correctness": json.loads(
            (args.ppl_summary.parents[1] / "correctness.json").read_text()
        ),
        "spearman": {
            "overall": spearman["overall"],
            "by_layer": spearman["by_layer"],
        },
        "ppl": {
            "config": ppl["config"],
            "aggregate": ppl["aggregate"],
            "matched_budget_nll_interpolation": matched,
            "interpolation_note": (
                "Piecewise-linear interpolation in NLL over measured causal budget."
            ),
        },
    }
    report["ppl"]["config"]["output_dir"] = str(
        report["ppl"]["config"]["output_dir"]
    )
    report["ppl"]["config"]["output_root"] = str(
        report["ppl"]["config"]["output_root"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(matched, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
