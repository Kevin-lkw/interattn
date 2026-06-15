import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .common import DEFAULT_OUTPUT_ROOT


METHODS = [
    ("condition_block", "Condition-block", "o", "#2563eb"),
    ("attention_topk", "Oracle attention top-k", "s", "#dc2626"),
    ("h2o", "H2O", "^", "#16a34a"),
    ("quest", "QUEST", "D", "#9333ea"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot aggregate multi-sample PPL for all methods."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-samples",
        type=int,
        default=100,
        help="Require at least this many completed samples for every method.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.output_root / "all_methods_mean_ppl_vs_budget.png"
    loaded = []
    for method, label, marker, color in METHODS:
        path = args.output_root / method / "summary.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")
        summary = torch.load(path, map_location="cpu", weights_only=False)
        completed = int(summary["aggregate"]["num_completed_samples"])
        if completed < args.require_samples:
            raise ValueError(
                f"{method} has only {completed} completed samples; "
                f"{args.require_samples} required."
            )
        points = sorted(
            (
                float(record["mean_measured_budget"]),
                float(record["mean_ppl"]),
                float(setting),
            )
            for setting, record in summary["aggregate"]["settings"].items()
        )
        loaded.append((label, marker, color, points, summary))

    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    for label, marker, color, points, _summary in loaded:
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=marker,
            linewidth=1.7,
            label=label,
            color=color,
        )

    teacher = loaded[0][4]["aggregate"].get("teacher")
    if teacher:
        ax.axhline(
            float(teacher["mean_ppl"]),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            label="Full-attention mean PPL",
        )
    ax.set_title(f"WikiText-2: mean PPL over {args.require_samples} samples")
    ax.set_xlabel("Mean equivalent causal attention budget")
    ax.set_ylabel("Mean sample PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.24)
    ax.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved combined plot to: {output}")

    for label, _marker, _color, points, _summary in loaded:
        print(f"\n{label}")
        print("setting\tmean_budget\tmean_ppl")
        for budget, ppl, setting in points:
            print(f"{setting:g}\t{budget:.6f}\t{ppl:.6f}")


if __name__ == "__main__":
    main()
