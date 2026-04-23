"""
Run baseline and optimal routing online, and compare with inter-optV routing.

Inter-optV pipeline per layer (avgV-only):
1) build H2O-with-belong mask
2) compute full-attention alpha (oracle)
3) for each HH cluster, use tildeV=avgV and solve optimal scalar r*
4) add recent-window contribution, patch hidden states, and run multilayer evaluation

Note:
- This runner is deterministic (no trainable parameters).
- Strategy is restricted to h2o because optV clustering depends on belong metadata.
"""

import os
import time

import torch
from torch.nn import functional as F

from .attention import gen_mask_h2o_with_belong, get_attention_map_after_rope
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .runner import (
    load_context,
    normalize_budget_key,
    resolve_layers,
    validate_args_with_cache,
)
from .runner_utils import (
    build_context_with_new_start,
    create_base_runner_parser,
    finalize_runner_args,
    load_existing_baseline_metrics,
    load_existing_optimal_metrics,
    mean_nll_and_ppl,
    resolve_head_indices,
    resolve_output_dir,
    set_seed,
    str_to_torch_dtype,
)
from .sanity import get_tail_labels, move_model_inputs_to_device


def parse_args():
    parser = create_base_runner_parser(
        description=(
            "Run baseline/optimal/inter_optV routing online and compare PPL. "
            "Inter-optV uses per-cluster optimal scalar with tildeV=avgV per layer."
        ),
        strategy_choices=["h2o"],
        default_strategy="h2o",
        strategy_help="optV refinement currently requires h2o_with_belong metadata.",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "For inter-optV this only changes evaluation sample; no fitted params are reused."
        ),
    )
    parser.add_argument(
        "--tilde-v",
        type=str,
        default="avg",
        choices=["avg", "hh"],
        help="Cluster representative tilde_V used by optV (avg or hh).",
    )
    return finalize_runner_args(parser.parse_args())


def _compute_full_attention_alpha(ctx, layer_idx, head_idx, pos_list, device):
    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    qk_sel = qk_scores[head_idx][:, pos_list, :].to(torch.float32)

    n_pos, seq_len = len(pos_list), qk_sel.shape[-1]
    causal = torch.full((n_pos, seq_len), float("-inf"), device=device)
    for i, pos in enumerate(pos_list):
        causal[i, : pos + 1] = 0.0

    return F.softmax(qk_sel + causal.unsqueeze(0), dim=-1)


def _extract_hh_positions(mask, recent_budget, pos_list):
    n_heads, _, _ = mask.shape
    hh_positions = []
    for h in range(n_heads):
        head_hh = []
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            recent_start = max(0, total_available - recent_budget)
            kept = ~torch.isneginf(mask[h, i, :total_available])
            kept[recent_start:total_available] = False
            head_hh.append(kept.nonzero(as_tuple=False).squeeze(-1).tolist())
        hh_positions.append(head_hh)
    return hh_positions


def _build_hh_mask(hh_positions, device):
    n_heads = len(hh_positions)
    n_pos = len(hh_positions[0]) if n_heads > 0 else 0
    max_hh = max(
        (len(hh_positions[h][i]) for h in range(n_heads) for i in range(n_pos)),
        default=0,
    )

    hh_mask = torch.zeros(n_heads, n_pos, max_hh, dtype=torch.bool, device=device)
    for h in range(n_heads):
        for i in range(n_pos):
            n = len(hh_positions[h][i])
            if n > 0:
                hh_mask[h, i, :n] = True
    return hh_mask


def _canonicalize_belong(belong, pos_list):
    root = belong.clone()
    n_heads, n_pos, _ = root.shape

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            row = root[h, i, :total_available]

            while True:
                parent = row[row]
                if torch.equal(parent, row):
                    break
                row = parent

            root[h, i, :total_available] = row

    return root


def _compute_cluster_gt(alpha_full, v_head, belong_root, hh_positions):
    n_heads, n_pos, _ = alpha_full.shape
    d = v_head.shape[-1]
    max_hh = max(
        (len(hh_positions[h][i]) for h in range(n_heads) for i in range(n_pos)),
        default=0,
    )

    if max_hh == 0:
        z1 = torch.zeros(n_heads, n_pos, 0, d, device=alpha_full.device)
        z2 = torch.zeros(n_heads, n_pos, 0, device=alpha_full.device)
        return z1, z2

    g = torch.zeros(n_heads, n_pos, max_hh, d, device=alpha_full.device)
    cluster_w = torch.zeros(n_heads, n_pos, max_hh, device=alpha_full.device)

    for h in range(n_heads):
        for i in range(n_pos):
            hh_ids = hh_positions[h][i]
            for slot, hh in enumerate(hh_ids):
                members = (belong_root[h, i] == hh).nonzero(as_tuple=False).squeeze(-1)
                if members.numel() == 0:
                    continue
                w = alpha_full[h, i, members].float()
                vm = v_head[h, members].float()
                g[h, i, slot] = (w.unsqueeze(-1) * vm).sum(0)
                cluster_w[h, i, slot] = w.sum()

    return g, cluster_w


def _build_tilde_v(v_head, belong_root, hh_positions, g, mode):
    n_heads, n_pos, n_hh, d = g.shape
    tv = torch.zeros(n_heads, n_pos, n_hh, d, device=v_head.device)

    for h in range(n_heads):
        for i in range(n_pos):
            hh_ids = hh_positions[h][i]
            for slot, hh in enumerate(hh_ids):
                if mode == "hh":
                    tv[h, i, slot] = v_head[h, hh].float()
                else:
                    members = (belong_root[h, i] == hh).nonzero(as_tuple=False).squeeze(-1)
                    if members.numel() == 0:
                        tv[h, i, slot] = v_head[h, hh].float()
                    else:
                        tv[h, i, slot] = v_head[h, members].float().mean(0)

    return tv


def _optimal_scalar_output(tilde_v, g, hh_mask):
    tv_norm2 = (tilde_v * tilde_v).sum(-1).clamp_min(1e-12)
    dot = (tilde_v * g).sum(-1)
    r_star = (dot / tv_norm2) * hh_mask.float()

    out_cluster = (r_star.unsqueeze(-1) * tilde_v).sum(-2)
    residual = torch.norm(r_star.unsqueeze(-1) * tilde_v - g, p=2, dim=-1)
    return out_cluster, r_star, residual


def _compute_recent_contrib(alpha_full, v_head, pos_list, recent_budget):
    n_heads, n_pos, _ = alpha_full.shape
    d = v_head.shape[-1]
    out = torch.zeros(n_heads, n_pos, d, device=alpha_full.device)

    for i, pos in enumerate(pos_list):
        total_available = pos + 1
        recent_start = max(0, total_available - recent_budget)
        if recent_start >= total_available:
            continue
        idx = torch.arange(recent_start, total_available, device=alpha_full.device)
        w = alpha_full[:, i, idx].float()
        v = v_head[:, idx, :].float()
        out[:, i] = (w.unsqueeze(-1) * v).sum(1)

    return out


def _build_modified_attn_hidden_from_vnew(ctx, layer_idx, head_idx, pos_list, v_new, device=None):
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    layer = ctx.model.model.layers[layer_idx]
    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)

    output = original.clone()
    output[head_idx] = v_new.to(output.dtype)

    attn_hidden = layer.self_attn.o_proj(output.permute(1, 0, 2).reshape(len(pos_list), -1))
    return attn_hidden


def _build_inter_optv_patches(
    ctx,
    args,
    budget,
    layer_idx_list,
    head_idx,
    pos_list,
    model_inputs,
):
    patches = {}
    layer_meta = {}
    build_start = time.time()
    total_layers = len(layer_idx_list)

    for layer_ord, layer_idx in enumerate(layer_idx_list, start=1):
        layer_start = time.time()
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

        route_mask, belong, _count = gen_mask_h2o_with_belong(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            budget=budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )

        alpha_full = _compute_full_attention_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            device=ctx.device,
        )

        v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()
        v_gt = (
            layer_ctx.attn_output[layer_idx]["output"][0, pos_list]
            .permute(1, 0, 2)
            .to(ctx.device)[head_idx]
            .float()
        )

        visible = int(args.seq_len * budget)
        if args.adaptive_budget and (layer_idx == 0 or layer_idx == 1):
            visible = args.seq_len
        recent_budget = visible // 2

        hh_positions = _extract_hh_positions(route_mask, recent_budget, pos_list)
        hh_mask = _build_hh_mask(hh_positions, ctx.device)
        belong_root = _canonicalize_belong(belong, pos_list)

        g, cluster_w = _compute_cluster_gt(
            alpha_full=alpha_full,
            v_head=v_head,
            belong_root=belong_root,
            hh_positions=hh_positions,
        )
        tilde_v = _build_tilde_v(
            v_head=v_head,
            belong_root=belong_root,
            hh_positions=hh_positions,
            g=g,
            mode=args.tilde_v,
        )

        out_cluster, r_star, residual = _optimal_scalar_output(tilde_v, g, hh_mask)
        recent_gt = _compute_recent_contrib(
            alpha_full=alpha_full,
            v_head=v_head,
            pos_list=pos_list,
            recent_budget=recent_budget,
        )
        v_optv = out_cluster + recent_gt

        patches[layer_idx] = _build_modified_attn_hidden_from_vnew(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            v_new=v_optv,
            device=ctx.device,
        )

        v_l2 = torch.norm(v_optv - v_gt, p=2, dim=-1).mean(dim=0)
        vals = r_star[hh_mask]
        r_mean = float(vals.mean().item()) if vals.numel() > 0 else float("nan")
        r_std = float(vals.std().item()) if vals.numel() > 0 else float("nan")

        valid_residual = residual * hh_mask.float()
        n_hh_per_pos = hh_mask.float().sum(-1)
        mean_residual = (valid_residual.sum(-1) / n_hh_per_pos.clamp_min(1.0)).mean()

        layer_elapsed = time.time() - layer_start
        total_elapsed = time.time() - build_start
        avg_layer_time = total_elapsed / float(layer_ord)
        remain_layers = total_layers - layer_ord
        eta_seconds = avg_layer_time * remain_layers

        print(
            f"[inter_optV][budget={budget}] layer {layer_idx} "
            f"({layer_ord}/{total_layers}) done | "
            f"layer_time={layer_elapsed:.2f}s elapsed={total_elapsed:.2f}s "
            f"eta={eta_seconds:.2f}s"
        )

        layer_meta[layer_idx] = {
            "tilde_v": args.tilde_v,
            "recent_budget": int(recent_budget),
            "visible_budget": int(visible),
            "mean_v_l2": float(v_l2.mean().item()),
            "mean_r_star": r_mean,
            "std_r_star": r_std,
            "mean_cluster_residual": float(mean_residual.item()),
            "mean_cluster_weight": float(cluster_w.mean().item()) if cluster_w.numel() > 0 else 0.0,
            "layer_time_sec": float(layer_elapsed),
            "eta_after_layer_sec": float(eta_seconds),
        }

    return patches, layer_meta


def main():
    set_seed(42)
    args = parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    dtype = str_to_torch_dtype(args.dtype)

    fit_ctx = load_context(args, dtype=dtype, device=args.device)
    validate_args_with_cache(fit_ctx, args)
    fit_ctx.model.eval()

    eval_ctx = fit_ctx
    if args.eval_start != args.start:
        eval_ctx = build_context_with_new_start(
            base_ctx=fit_ctx,
            dataset_name=args.dataset,
            start=args.eval_start,
            seq_len=args.seq_len,
        )
        validate_args_with_cache(eval_ctx, args)

    head_idx = resolve_head_indices(fit_ctx.model_config.num_attention_heads)
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        fit_ctx.model_config.num_hidden_layers,
    )

    eval_model_inputs = move_model_inputs_to_device(eval_ctx.inputs, eval_ctx.device)
    eval_labels = get_tail_labels(eval_ctx, pos_list, eval_ctx.device)

    with torch.no_grad():
        ref_logits = eval_ctx.model(**eval_model_inputs, use_cache=False).logits[:, pos_list, :].float()
    ref_nll, ref_ppl = mean_nll_and_ppl(ref_logits, eval_labels)

    out_dir = resolve_output_dir(args, runner_name="runner_inter_optV")
    save_path = os.path.join(out_dir, f"runner_inter_optV_{args.tilde_v}_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "method": f"inter_optV_{args.tilde_v}V",
        "tilde_v": args.tilde_v,
        "budgets": [float(x) for x in args.budgets],
        "layers": layer_idx_list,
        "reference": {"nll": ref_nll, "ppl": ref_ppl},
        "results": {},
    }

    if os.path.exists(save_path):
        old_summary = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(old_summary, dict):
            old_results = old_summary.get("results", {})
            if isinstance(old_results, dict):
                summary["results"].update(old_results)
            for key in [
                "dataset",
                "fit_start",
                "eval_start",
                "seq_len",
                "strategy",
                "loss_type",
                "method",
                "tilde_v",
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing runner_inter_optV summary: {save_path}")

    pending_budgets = []
    for budget in args.budgets:
        if normalize_budget_key(summary["results"], budget) is None:
            pending_budgets.append(float(budget))
        else:
            print(f"[skip] budget={budget} already exists in summary, skip.")

    if len(pending_budgets) == 0:
        print("No pending budgets to run. Keep existing summary unchanged.")
        print(f"Summary path: {save_path}")
        return

    for budget in pending_budgets:
        print(f"\n[runner_inter_optV] budget={budget}")

        baseline_loaded = None
        optimal_loaded = None
        baseline_nll = baseline_ppl = None
        optimal_nll = optimal_ppl = None
        try:
            baseline_loaded = load_existing_baseline_metrics(args=args, budget=budget)
            optimal_loaded = load_existing_optimal_metrics(args=args, budget=budget)
            baseline_nll, baseline_ppl = baseline_loaded["nll"], baseline_loaded["ppl"]
            optimal_nll, optimal_ppl = optimal_loaded["nll"], optimal_loaded["ppl"]
            print(
                f"[loaded metrics@eval_start={args.eval_start}] "
                f"baseline_nll={baseline_nll:.6f}, baseline_ppl={baseline_ppl:.6f}, "
                f"optimal_nll={optimal_nll:.6f}, optimal_ppl={optimal_ppl:.6f}"
            )
        except (FileNotFoundError, KeyError) as exc:
            print(
                "[loaded metrics] skipped baseline/optimal loading for eval sample "
                f"(eval_start={args.eval_start}): {exc}"
            )

        inter_optv_patches, inter_optv_meta = _build_inter_optv_patches(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )

        inter_optv_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_optv_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_optv_nll, inter_optv_ppl = mean_nll_and_ppl(inter_optv_logits, eval_labels)

        delta_vs_ref = {
            "inter_optv_nll_gap": inter_optv_nll - ref_nll,
        }
        if baseline_nll is not None:
            delta_vs_ref["baseline_nll_gap"] = baseline_nll - ref_nll
        if optimal_nll is not None:
            delta_vs_ref["optimal_nll_gap"] = optimal_nll - ref_nll

        loaded_metrics = {}
        if baseline_loaded is not None:
            loaded_metrics["baseline"] = baseline_loaded["raw"]
        if optimal_loaded is not None:
            loaded_metrics["optimal"] = optimal_loaded["raw"]

        summary["results"][float(budget)] = {
            "baseline": None if baseline_nll is None else {"nll": baseline_nll, "ppl": baseline_ppl},
            "optimal": None if optimal_nll is None else {"nll": optimal_nll, "ppl": optimal_ppl},
            "inter_optv": {"nll": inter_optv_nll, "ppl": inter_optv_ppl},
            "delta_vs_ref": delta_vs_ref,
            "meta": inter_optv_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, inter_optV={inter_optv_ppl:.6f}"
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
