"""Line graph: PPL vs measured budget for cosh vs Bennett condition (multisample
aggregates), plus the previous saved baseline. Usage:

  python plot_ppl_curves.py --curves label=path/to/summary.pt ... --out plot.png
"""

import argparse

import matplotlib.pyplot as plt
import torch

COLORS = ["#6b7280", "#2563eb", "#dc2626", "#059669", "#d97706"]


def load_curve(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    agg = s["aggregate"]
    pts = sorted(
        (a["mean_measured_budget"], a["mean_ppl"], eps)
        for eps, a in agg["settings"].items()
        if a.get("count", 0) > 0
    )
    return pts, float(agg["teacher"]["mean_ppl"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="+", required=True, help="label=summary.pt entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="PPL vs budget: cosh vs Bennett condition")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    teacher = None
    for i, entry in enumerate(args.curves):
        label, path = entry.split("=", 1)
        pts, t = load_curve(path)
        teacher = t if teacher is None else teacher
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.5,
                color=COLORS[i % len(COLORS)], label=label)
        for x, y, eps in pts:
            ax.annotate(f"{eps:g}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=COLORS[i % len(COLORS)])
    if teacher is not None:
        ax.axhline(teacher, color="#111827", linestyle="--", linewidth=1.0)
        ax.text(0.99, teacher, f"teacher {teacher:.3f}", va="bottom", ha="right",
                fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("measured budget (causal fraction)")
    ax.set_ylabel("mean PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.24)
    ax.legend(fontsize=9)
    ax.set_title(args.title)
    fig.savefig(args.out, dpi=180)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
