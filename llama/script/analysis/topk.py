import math
import os

import matplotlib.pyplot as plt
import torch

from .config import parse_args, set_seed, str_to_torch_dtype
from .online_routing import capture_layer_artifacts
from .runner import load_context, resolve_layers
from .sanity import move_model_inputs_to_device


def compute_attention_probs_from_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q/k: [B, H, S, D]
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])  # [B, H, S, S]

    seq_len = scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return torch.softmax(scores, dim=-1)


def topk_mass_distribution(attn_probs: torch.Tensor, budget: float) -> torch.Tensor:
    # attn_probs: [B, H, S, S]
    seq_len = attn_probs.shape[-1]
    visible = int(seq_len * budget)
    if visible <= 0:
        raise ValueError(f"budget={budget} gives visible={visible}, increase budget or seq_len")

    masses = []
    for pos in range(seq_len):
        available = pos + 1
        rows = attn_probs[:, :, pos, :available].reshape(-1, available)
        if available <= visible:
            mass = torch.ones(rows.shape[0], dtype=torch.float32, device=attn_probs.device)
        else:
            mass = rows.topk(k=visible, dim=-1).values.sum(dim=-1)
        masses.append(mass)

    return torch.cat(masses, dim=0)


def plot_hist(values: torch.Tensor, budget: float, output_path: str):
    values_np = values.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(values_np, bins=50, range=(0.0, 1.0), alpha=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Top-k attention score mass")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-k score mass distribution (budget={budget:g})")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    if args.strategy != "attention_topk":
        print(f"[INFO] strategy={args.strategy} is ignored in this script; analyzing attention topk mass only.")

    ctx = load_context(args, dtype=dtype, device=args.device)
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    ctx.model.eval()

    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )
    pos_list = list(range(args.seq_len))

    budget_to_values = {float(b): [] for b in args.budgets}
    output_dir = f"../result/{args.dataset}_{args.start}/attention_topk_analysis"
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx in layer_idx_list:
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=None,
        )

        q = artifacts["q"].to(ctx.device)  # [B, H, S, D]
        k = artifacts["k"].to(ctx.device)  # [B, H, S, D]

        attn_probs = compute_attention_probs_from_qk(q, k)

        for budget in args.budgets:
            b = float(budget)
            masses = topk_mass_distribution(attn_probs, b)
            budget_to_values[b].append(masses.detach().cpu())

            # Extra output: one histogram per (budget, layer).
            output_dir = f"../result/{args.dataset}_{args.start}/attention_topk_analysis/budget_{b:g}/layer_{layer_idx}"
            os.makedirs(output_dir, exist_ok=True)
            layer_output_path = os.path.join(
                output_dir,
                f"topk_mass_hist_budget_{b:g}_layer_{layer_idx}.png",
            )
            plot_hist(masses, b, layer_output_path)

        print(f"[done] layer {layer_idx}")

    for budget in args.budgets:
        b = float(budget)
        all_values = torch.cat(budget_to_values[b], dim=0)
        output_dir = f"../result/{args.dataset}_{args.start}/attention_topk_analysis/budget_{b:g}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"topk_mass_hist_budget_{b:g}.png")
        plot_hist(all_values, b, output_path)

        print(
            f"budget={b:g}: count={all_values.numel()}, mean={all_values.mean().item():.6f}, "
            f"std={all_values.std(unbiased=False).item():.6f}"
        )
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
