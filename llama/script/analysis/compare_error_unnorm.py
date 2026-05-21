"""
Decompose the count-refined H2O-all routing error in unnormalized numerator space.

This script intentionally ignores every softmax denominator.  For each query row it
compares shifted attention numerators under the same row-wise max shift:

    count_all_num       = S_tilde
    key_corrected_num   = S_tilde + key_num_error
    value_corrected_num = S_tilde + value_num_error
    reconstructed_num   = S_tilde + key_num_error + value_num_error
    full_num            = S_full

The reconstructed numerator should match full_num up to numerical precision.
"""

import argparse
import os
import warnings

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong, get_qk_logits
from .compare_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    plot_per_pos_two_lines,
    resolve_head_indices,
    resolve_output_dir,
    validate_common_args,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose count-all routing error in unnormalized numerator space."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    parser.add_argument(
        "--merge-metric",
        choices=["k", "v"],
        default="k",
        help="Cluster merge target metric. Default is k; v is allowed only for diagnostics.",
    )
    return parser.parse_args()


def progress_iter(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def decompose_count_all_error_unnorm(qk_logits, v_head, route_mask, belong_root, count, pos_list):
    n_heads, n_pos, seq_len = qk_logits.shape
    d = v_head.shape[-1]
    if route_mask.shape != qk_logits.shape:
        raise ValueError(
            f"route_mask shape mismatch: got {tuple(route_mask.shape)} expected {tuple(qk_logits.shape)}"
        )
    if belong_root.shape != qk_logits.shape:
        raise ValueError(
            f"belong_root shape mismatch: got {tuple(belong_root.shape)} expected {tuple(qk_logits.shape)}"
        )
    if count.shape != qk_logits.shape:
        raise ValueError(f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}")
    if v_head.shape[0] != n_heads or v_head.shape[1] != seq_len:
        raise ValueError(
            f"v_head shape mismatch: got {tuple(v_head.shape)} expected ({n_heads}, {seq_len}, d)"
        )

    count_all_num = torch.zeros(n_heads, n_pos, d, device=qk_logits.device)
    key_num_error = torch.zeros_like(count_all_num)
    value_num_error = torch.zeros_like(count_all_num)
    full_num = torch.zeros_like(count_all_num)

    total_steps = n_heads * n_pos
    outer = ((h, i, pos) for h in range(n_heads) for i, pos in enumerate(pos_list))
    for h, i, pos in progress_iter(
        outer,
        total=total_steps,
        desc="decompose unnorm clusters",
        dynamic_ncols=True,
    ):
        total_available = pos + 1
        row_logits = qk_logits[h, i, :total_available].float()
        row_shift = row_logits.max()
        row_exp = torch.exp(row_logits - row_shift)
        row_v = v_head[h, :total_available].float()
        row_root = belong_root[h, i, :total_available]
        if (row_root < 0).any():
            raise ValueError(f"belong_root contains negative index at head={h}, pos={pos}")

        kept = (~torch.isneginf(route_mask[h, i, :total_available])).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if kept.numel() == 0:
            raise ValueError(f"No kept representatives at head={h}, pos={pos}")

        c = count[h, i, kept].to(torch.float32)
        if (c <= 0).any():
            raise ValueError(f"Kept representative has non-positive count at head={h}, pos={pos}")

        cluster_exp = torch.zeros(total_available, device=qk_logits.device, dtype=torch.float32)
        cluster_exp.scatter_add_(0, row_root, row_exp)

        cluster_count = torch.zeros(total_available, device=qk_logits.device, dtype=torch.long)
        cluster_count.scatter_add_(0, row_root, torch.ones_like(row_root, dtype=torch.long))
        if not torch.equal(cluster_count[kept], count[h, i, kept]):
            raise ValueError(f"count/belong mismatch at head={h}, pos={pos}")

        rep_exp = row_exp[kept]
        v_rep = v_head[h, kept].float()
        weighted_rep_exp = c * rep_exp

        count_all_num[h, i] = (weighted_rep_exp.unsqueeze(-1) * v_rep).sum(0)

        key_exp_gap = cluster_exp[kept] - weighted_rep_exp
        key_num_error[h, i] = (key_exp_gap.unsqueeze(-1) * v_rep).sum(0)

        full_num[h, i] = (row_exp.unsqueeze(-1) * row_v).sum(0)
        value_num_error[h, i] = full_num[h, i] - count_all_num[h, i] - key_num_error[h, i]

    key_corrected_num = count_all_num + key_num_error
    value_corrected_num = count_all_num + value_num_error
    reconstructed_num = count_all_num + key_num_error + value_num_error

    return {
        "count_all_num": count_all_num,
        "key_corrected_num": key_corrected_num,
        "value_corrected_num": value_corrected_num,
        "reconstructed_num": reconstructed_num,
        "full_num": full_num,
        "key_num_error": key_num_error,
        "value_num_error": value_num_error,
    }


def l2_per_pos(x, target):
    return torch.norm(x.float() - target.float(), p=2, dim=-1).mean(dim=0)


def norm_per_pos(x):
    return torch.norm(x.float(), p=2, dim=-1).mean(dim=0)


def cosine_per_pos(x, target, eps=1e-12):
    x_f = x.float()
    target_f = target.float()
    dot = (x_f * target_f).sum(dim=-1)
    denom = torch.norm(x_f, p=2, dim=-1) * torch.norm(target_f, p=2, dim=-1)
    return (dot / denom.clamp_min(eps)).mean(dim=0)


def save_unnorm_decomposition_tsv(out_path, pos_list, metrics):
    names = [
        "count_all_num_l2",
        "value_corrected_num_l2",
        "key_corrected_num_l2",
        "reconstructed_num_l2",
        "count_all_full_num_cos",
        "value_corrected_full_num_cos",
        "key_corrected_full_num_cos",
        "reconstructed_full_num_cos",
        "key_num_error_l2",
        "value_num_error_l2",
        "key_delta_l2",
        "value_delta_l2",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("pos\t" + "\t".join(names) + "\n")
        for i, pos in enumerate(pos_list):
            vals = [float(metrics[name][i].item()) for name in names]
            f.write(f"{pos}\t" + "\t".join(f"{v:.8e}" for v in vals) + "\n")


def main():
    set_seed(42)
    args = parse_args()
    if args.merge_metric == "v":
        warnings.warn(
            "--merge-metric v is not the intended setting for this decomposition; "
            "use the default --merge-metric k for the main analysis.",
            stacklevel=2,
        )

    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_common_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )

    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty pos_list after applying --pos-start/--pos-end")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(
        args=args,
        head_idx=head_idx,
        compare_tag="compare_error_unnorm",
        include_loss_type=True,
    )

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
    else:
        prefix_patches = build_baseline_prefix_patches(
            ctx=ctx,
            args=args,
            target_layer=args.layer,
            pos_list=pos_list,
            model_inputs=model_inputs,
            build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask_h2o_with_belong_all(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                head_idx=hi,
                budget=args.budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
                merge_metric=args.merge_metric,
            )[0],
        )

    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong, count = gen_mask_h2o_with_belong_all(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        merge_metric=args.merge_metric,
    )
    belong_root = canonicalize_belong(belong, pos_list)

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )
    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()

    decomp = decompose_count_all_error_unnorm(
        qk_logits=qk_logits,
        v_head=v_head,
        route_mask=route_mask,
        belong_root=belong_root,
        count=count,
        pos_list=pos_list,
    )

    full_num = decomp["full_num"]
    metrics = {
        "count_all_num_l2": l2_per_pos(decomp["count_all_num"], full_num),
        "value_corrected_num_l2": l2_per_pos(decomp["value_corrected_num"], full_num),
        "key_corrected_num_l2": l2_per_pos(decomp["key_corrected_num"], full_num),
        "reconstructed_num_l2": l2_per_pos(decomp["reconstructed_num"], full_num),
        "count_all_full_num_cos": cosine_per_pos(decomp["count_all_num"], full_num),
        "value_corrected_full_num_cos": cosine_per_pos(decomp["value_corrected_num"], full_num),
        "key_corrected_full_num_cos": cosine_per_pos(decomp["key_corrected_num"], full_num),
        "reconstructed_full_num_cos": cosine_per_pos(decomp["reconstructed_num"], full_num),
        "key_num_error_l2": norm_per_pos(decomp["key_num_error"]),
        "value_num_error_l2": norm_per_pos(decomp["value_num_error"]),
        "key_delta_l2": norm_per_pos(decomp["key_corrected_num"] - decomp["count_all_num"]),
        "value_delta_l2": norm_per_pos(decomp["value_corrected_num"] - decomp["count_all_num"]),
    }

    recon_abs = torch.norm(decomp["reconstructed_num"] - full_num, p=2, dim=-1)
    print("===== Compare-Error-Unnorm Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}, merge_metric={args.merge_metric}"
    )
    print(
        f"mean count_all num_l2={float(metrics['count_all_num_l2'].mean().item()):.8e}, "
        f"mean value_corrected num_l2={float(metrics['value_corrected_num_l2'].mean().item()):.8e}, "
        f"mean key_corrected num_l2={float(metrics['key_corrected_num_l2'].mean().item()):.8e}, "
        f"mean reconstructed num_l2={float(metrics['reconstructed_num_l2'].mean().item()):.8e}"
    )
    print(
        f"mean key_delta_l2={float(metrics['key_delta_l2'].mean().item()):.8e}, "
        f"mean value_delta_l2={float(metrics['value_delta_l2'].mean().item()):.8e}, "
        f"sanity max reconstructed-full_num l2={float(recon_abs.max().item()):.8e}"
    )
    print(
        f"mean count_all full_num_cos={float(metrics['count_all_full_num_cos'].mean().item()):.8e}, "
        f"mean value_corrected full_num_cos={float(metrics['value_corrected_full_num_cos'].mean().item()):.8e}, "
        f"mean key_corrected full_num_cos={float(metrics['key_corrected_full_num_cos'].mean().item()):.8e}, "
        f"mean reconstructed full_num_cos={float(metrics['reconstructed_full_num_cos'].mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_error_unnorm_decomposition.tsv")
    save_unnorm_decomposition_tsv(per_pos_path, pos_list, metrics)

    count_vs_key_plot = os.path.join(output_dir, "per_pos_count_all_vs_key_corrected_unnorm.png")
    plot_per_pos_two_lines(
        out_path=count_vs_key_plot,
        pos_list=pos_list,
        y1=metrics["count_all_num_l2"],
        y2=metrics["key_corrected_num_l2"],
        label1="count_all_num_l2",
        label2="key_corrected_num_l2",
        title="Per-Position Numerator L2: Count-All vs Key-Corrected",
        dpi=args.plot_dpi,
    )

    term_plot = os.path.join(output_dir, "per_pos_key_vs_value_delta_unnorm.png")
    plot_per_pos_two_lines(
        out_path=term_plot,
        pos_list=pos_list,
        y1=metrics["key_delta_l2"],
        y2=metrics["value_delta_l2"],
        label1="key_delta_l2",
        label2="value_delta_l2",
        title="Per-Position Numerator Delta Norm: Key vs Value",
        dpi=args.plot_dpi,
    )

    belong_path = os.path.join(output_dir, "belong.pt")
    torch.save(belong.detach().cpu(), belong_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "metric_name": "unnormalized_numerator_l2",
        "mean_count_all_num_l2": float(metrics["count_all_num_l2"].mean().item()),
        "mean_value_corrected_num_l2": float(metrics["value_corrected_num_l2"].mean().item()),
        "mean_key_corrected_num_l2": float(metrics["key_corrected_num_l2"].mean().item()),
        "mean_reconstructed_num_l2": float(metrics["reconstructed_num_l2"].mean().item()),
        "mean_count_all_full_num_cos": float(metrics["count_all_full_num_cos"].mean().item()),
        "mean_value_corrected_full_num_cos": float(metrics["value_corrected_full_num_cos"].mean().item()),
        "mean_key_corrected_full_num_cos": float(metrics["key_corrected_full_num_cos"].mean().item()),
        "mean_reconstructed_full_num_cos": float(metrics["reconstructed_full_num_cos"].mean().item()),
        "mean_key_delta_l2": float(metrics["key_delta_l2"].mean().item()),
        "mean_value_delta_l2": float(metrics["value_delta_l2"].mean().item()),
        "mean_key_num_error_l2": float(metrics["key_num_error_l2"].mean().item()),
        "mean_value_num_error_l2": float(metrics["value_num_error_l2"].mean().item()),
        "sanity_mean_reconstructed_full_num_l2": float(recon_abs.mean().item()),
        "sanity_max_reconstructed_full_num_l2": float(recon_abs.max().item()),
        "belong": belong.detach().cpu(),
        "belong_root": belong_root.detach().cpu(),
        "count": count.detach().cpu(),
    }
    for name, value in metrics.items():
        stats[f"{name}_per_pos"] = value.detach().cpu()

    stats_path = os.path.join(output_dir, "compare_error_unnorm_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos unnorm decomposition table to: {per_pos_path}")
    print(f"Saved count/key unnorm plot to: {count_vs_key_plot}")
    print(f"Saved key/value unnorm delta plot to: {term_plot}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
