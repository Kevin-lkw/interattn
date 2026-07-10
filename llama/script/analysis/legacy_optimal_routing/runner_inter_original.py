"""
Run baseline and optimal routing online, and compare with inter-original routing.

Inter-original routing pipeline per layer:
1) build mask for kept tokens
2) compute full-attention alpha (oracle)
3) zero out dropped tokens without renormalizing
4) patch hidden states with alpha_original @ V and run multilayer evaluation

Note:
- This runner is deterministic (no trainable parameters).
- Supported strategies are limited to masks that define the kept tokens.
"""

import os

import torch
from torch.nn import functional as F

from ..attention import gen_mask, get_attention_map_after_rope
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
            "Run baseline/optimal/inter_original routing online and compare PPL. "
            "Inter-original uses full-attention weights on kept tokens (no renorm)."
        ),
        strategy_choices=["h2o", "attention_topk"],
        default_strategy="h2o",
        strategy_help="inter-original uses kept tokens from the selected mask.",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "For inter-original this only changes evaluation sample; no fitted params are reused."
        ),
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


def _build_inter_original_patches(
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

        route_mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
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

        keep_mask = (~torch.isneginf(route_mask)).float()
        alpha_original = alpha_full * keep_mask

        v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()
        v_original = alpha_original.float() @ v_head.float()

        patches[layer_idx] = _build_modified_attn_hidden_from_vnew(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            v_new=v_original,
            device=ctx.device,
        )

        visible = int(args.seq_len * budget)
        if args.adaptive_budget and (layer_idx == 0 or layer_idx == 1):
            visible = args.seq_len

        layer_meta[layer_idx] = {
            "visible_budget": int(visible),
            "mean_alpha": float(alpha_original.mean().item()),
            "mean_v_norm": float(torch.norm(v_original, p=2, dim=-1).mean().item()),
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

    out_dir = resolve_output_dir(args, runner_name="runner_inter_original")
    save_path = os.path.join(out_dir, "runner_inter_original_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "method": "inter_original",
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
        print(f"Loaded existing runner_inter_original summary: {save_path}")

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
        print(f"\n[runner_inter_original] budget={budget}")

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

        inter_orig_patches, inter_orig_meta = _build_inter_original_patches(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )

        inter_orig_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_orig_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_orig_nll, inter_orig_ppl = mean_nll_and_ppl(inter_orig_logits, eval_labels)

        delta_vs_ref = {
            "inter_original_nll_gap": inter_orig_nll - ref_nll,
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
            "inter_original": {"nll": inter_orig_nll, "ppl": inter_orig_ppl},
            "delta_vs_ref": delta_vs_ref,
            "meta": inter_orig_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = (
            f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, "
            f"inter_original={inter_orig_ppl:.6f}"
        )
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
