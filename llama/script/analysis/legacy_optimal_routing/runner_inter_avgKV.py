"""
Run baseline and optimal routing online, and compare with inter-avgKV routing.

Inter-avgKV routing pipeline per layer:
1) build baseline alpha from QK logits + H2O-with-belong mask
2) refine HH routing logits with q·avgK + log(C)
3) refine value vectors with avgV = sumV / C on HH representatives
4) patch hidden states with refined V and run multilayer evaluation

Note:
- This runner is deterministic (no trainable parameters).
- Strategy is restricted to h2o because avgKV refinement depends on belong/count metadata.
"""

import os

import torch
from torch.nn import functional as F

from ..attention import gen_mask_h2o_with_belong, get_attention_map_after_rope
from ..online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ..runtime import (
    load_context,
    resolve_layers,
    validate_args_with_cache,
)
from ..runner_utils import normalize_budget_key
from ..runner_utils import (
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
from ..sanity import get_tail_labels, move_model_inputs_to_device


def parse_args():
    parser = create_base_runner_parser(
        description=(
            "Run baseline/optimal/inter_avgKV routing online and compare PPL. "
            "Inter-avgKV routing uses q·avgK+log(C) and avgV refinement per layer."
        ),
        strategy_choices=["h2o"],
        default_strategy="h2o",
        strategy_help="avgKV refinement currently requires h2o_with_belong metadata.",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "For inter-avgKV this only changes evaluation sample; no fitted params are reused."
        ),
    )
    return finalize_runner_args(parser.parse_args())




def _get_qk_logits(ctx, layer_idx, head_idx, pos_list, device):
    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    return qk_scores[head_idx][:, pos_list, :].to(torch.float32)


def build_avgk_count_refined_alpha(
    qk_logits,
    q_head,
    mask,
    count,
    hh_sumk_idx,
    hh_sumk_val,
    pos_list,
    recent_budget,
):
    n_heads, n_pos, _seq_len = qk_logits.shape
    if mask.shape != qk_logits.shape:
        raise ValueError(
            f"mask shape mismatch: got {tuple(mask.shape)} expected {tuple(qk_logits.shape)}"
        )
    if count.shape != qk_logits.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}"
        )
    if q_head.shape[0] != n_heads or q_head.shape[1] != n_pos:
        raise ValueError(
            f"q_head shape mismatch: got {tuple(q_head.shape)} expected ({n_heads}, {n_pos}, d)"
        )

    logits = qk_logits.to(torch.float32).clone()
    mask_f = mask.to(torch.float32)
    scale = float(q_head.shape[-1]) ** 0.5

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            recent_start = max(0, total_available - recent_budget)

            idx = hh_sumk_idx[h][i]
            sumk = hh_sumk_val[h][i]
            if idx is None or sumk is None or idx.numel() == 0:
                continue

            visible = ~torch.isneginf(mask_f[h, i, :total_available])
            hh_visible = visible.clone()
            hh_visible[recent_start:total_available] = False
            if hh_visible[idx].any().item() is False:
                continue

            c = count[h, i, idx].to(torch.float32).clamp_min(1.0)
            avgk = sumk.float() / c.unsqueeze(-1)
            q = q_head[h, i].float().unsqueeze(0)
            q_avgk = (q * avgk).sum(dim=-1) / scale
            logits[h, i, idx] = q_avgk + torch.log(c)

    return F.softmax(logits + mask_f, dim=-1)


def build_avgv_refined_v(alpha, v_head, hh_sumv_idx, hh_sumv_val, count):
    n_heads, n_pos, seq_len = alpha.shape
    if v_head.shape[0] != n_heads or v_head.shape[1] != seq_len:
        raise ValueError(
            f"v_head shape mismatch: got {tuple(v_head.shape)} expected heads={n_heads}, seq={seq_len}, d=*"
        )
    if count.shape != alpha.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(alpha.shape)}"
        )

    v_new = alpha.float() @ v_head.float()
    for h in range(n_heads):
        for i in range(n_pos):
            idx = hh_sumv_idx[h][i]
            val_sum = hh_sumv_val[h][i]
            if idx is None or val_sum is None or idx.numel() == 0:
                continue

            c = count[h, i, idx].to(torch.float32).clamp_min(1.0).unsqueeze(-1)
            val_avg = val_sum.float() / c
            w = alpha[h, i, idx].float().unsqueeze(-1)
            delta = val_avg - v_head[h, idx].float()
            v_new[h, i] = v_new[h, i] + (w * delta).sum(dim=0)

    return v_new


def build_modified_attn_hidden_from_vnew(ctx, layer_idx, head_idx, pos_list, v_new, device=None):
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


def _build_inter_avgkv_patches(
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

    for layer_idx in layer_idx_list:
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

        route_mask, _belong, count, hh_sumv_idx, hh_sumv_val, hh_sumk_val = gen_mask_h2o_with_belong(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            budget=budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
            return_hh_sumv=True,
            return_hh_sumk=True,
        )

        qk_logits = _get_qk_logits(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            device=ctx.device,
        )

        q_head = (
            layer_ctx.rope_qkv[layer_idx]["q"]
            .to(ctx.device)[0][head_idx][:, pos_list, :]
            .float()
        )

        visible = int(args.seq_len * budget)
        if args.adaptive_budget and (layer_idx == 0 or layer_idx == 1):
            visible = args.seq_len
        recent_budget = visible // 2

        alpha_avgkv = build_avgk_count_refined_alpha(
            qk_logits=qk_logits,
            q_head=q_head,
            mask=route_mask,
            count=count,
            hh_sumk_idx=hh_sumv_idx,
            hh_sumk_val=hh_sumk_val,
            pos_list=pos_list,
            recent_budget=recent_budget,
        )

        v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()
        v_avgkv = build_avgv_refined_v(
            alpha=alpha_avgkv,
            v_head=v_head,
            hh_sumv_idx=hh_sumv_idx,
            hh_sumv_val=hh_sumv_val,
            count=count,
        )

        patches[layer_idx] = build_modified_attn_hidden_from_vnew(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            v_new=v_avgkv,
            device=ctx.device,
        )

        layer_meta[layer_idx] = {
            "recent_budget": int(recent_budget),
            "visible_budget": int(visible),
            "mean_alpha": float(alpha_avgkv.mean().item()),
            "mean_v_norm": float(torch.norm(v_avgkv, p=2, dim=-1).mean().item()),
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

    out_dir = resolve_output_dir(args, runner_name="runner_inter_avgKV")
    save_path = os.path.join(out_dir, "runner_inter_avgKV_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "method": "inter_avgKV",
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
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing runner_inter_avgKV summary: {save_path}")

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
        print(f"\n[runner_inter_avgKV] budget={budget}")

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

        inter_avgkv_patches, inter_avgkv_meta = _build_inter_avgkv_patches(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )

        inter_avgkv_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_avgkv_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_avgkv_nll, inter_avgkv_ppl = mean_nll_and_ppl(inter_avgkv_logits, eval_labels)

        delta_vs_ref = {
            "inter_avgkv_nll_gap": inter_avgkv_nll - ref_nll,
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
            "inter_avgkv": {"nll": inter_avgkv_nll, "ppl": inter_avgkv_ppl},
            "delta_vs_ref": delta_vs_ref,
            "meta": inter_avgkv_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, inter_avgKV={inter_avgkv_ppl:.6f}"
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
