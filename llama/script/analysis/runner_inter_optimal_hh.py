"""
Run baseline and optimal routing online, and compare with HH-only optimal routing.

Inter-optimal-HH pipeline per layer:
1) build baseline alpha from H2O QK routing
2) split visible keys into HH / recent
3) keep recent routing fixed to baseline
4) optimize HH-only routing on the remaining probability mass
5) patch hidden states with HH-only optimal alpha and run multilayer evaluation

Note:
- Strategy is restricted to h2o because HH/recent partition follows H2O.
- The learned alpha is sample-specific, so fitting is always done on the
  evaluation sample when --eval-start differs from --start.
"""

import os

import torch

from .attention import build_qk_routing_alpha, gen_mask
from .compare_optimal_routing_hh import (
    build_hh_recent_masks,
    optimize_alpha_hh_only,
    compute_recent_budget,
)
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
from .sanity import build_modified_attn_hidden, get_tail_labels, move_model_inputs_to_device


def parse_args():
    parser = create_base_runner_parser(
        description=(
            "Run baseline/optimal/inter_optimal_hh routing online and compare PPL. "
            "Inter-optimal-HH optimizes only HH routing while keeping recent routing fixed."
        ),
        strategy_choices=["h2o"],
        default_strategy="h2o",
        strategy_help="HH-only optimal routing currently requires h2o partition.",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "For inter_optimal_hh, fitting is always done on the evaluation sample."
        ),
    )
    return finalize_runner_args(parser.parse_args())


def build_inter_optimal_hh_patches(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs):
    patches = {}
    alpha_meta = {}
    losses_by_layer = {}

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

        alpha_base = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=route_mask,
            device=ctx.device,
        )

        recent_budget = compute_recent_budget(
            seq_len=args.seq_len,
            budget=budget,
            adaptive_budget=args.adaptive_budget,
            layer_idx=layer_idx,
        )
        hh_mask, recent_mask = build_hh_recent_masks(route_mask, pos_list, recent_budget)

        alpha_opt_hh, _p_alpha, _p_teacher, losses = optimize_alpha_hh_only(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            base_alpha=alpha_base,
            hh_mask=hh_mask,
            recent_mask=recent_mask,
            training_steps=args.training_steps,
            lr=args.lr,
            loss_type=args.loss_type,
            device=ctx.device,
        )

        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_opt_hh,
            device=ctx.device,
        )

        recent_diff = (alpha_opt_hh - alpha_base).abs() * recent_mask.float()
        max_recent_delta = float(recent_diff.max().item()) if recent_diff.numel() > 0 else 0.0
        hh_count = hh_mask.float().sum(dim=-1)
        hh_rows = hh_mask.any(dim=-1)
        avg_hh_count = float(hh_count[hh_rows].mean().item()) if torch.any(hh_rows) else 0.0

        alpha_meta[layer_idx] = {
            "recent_budget": int(recent_budget),
            "avg_hh_count": avg_hh_count,
            "max_recent_delta": max_recent_delta,
        }
        losses_by_layer[layer_idx] = losses

        print(
            f"[fit alpha_hh] layer={layer_idx} avg_hh_count={avg_hh_count:.2f} "
            f"max_recent_delta={max_recent_delta:.3e}"
        )

    return patches, alpha_meta, losses_by_layer


def main():
    set_seed(42)
    args = parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    dtype = str_to_torch_dtype(args.dtype)

    base_ctx = load_context(args, dtype=dtype, device=args.device)
    validate_args_with_cache(base_ctx, args)
    base_ctx.model.eval()

    eval_ctx = base_ctx
    if args.eval_start != args.start:
        eval_ctx = build_context_with_new_start(
            base_ctx=base_ctx,
            dataset_name=args.dataset,
            start=args.eval_start,
            seq_len=args.seq_len,
        )
        validate_args_with_cache(eval_ctx, args)
        print(
            "[runner_inter_optimal_hh] eval_start differs from start; "
            "fitting HH-only alpha on eval sample because alpha is sample-specific."
        )

    head_idx = resolve_head_indices(base_ctx.model_config.num_attention_heads)
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        base_ctx.model_config.num_hidden_layers,
    )

    eval_model_inputs = move_model_inputs_to_device(eval_ctx.inputs, eval_ctx.device)
    eval_labels = get_tail_labels(eval_ctx, pos_list, eval_ctx.device)

    with torch.no_grad():
        ref_logits = eval_ctx.model(**eval_model_inputs, use_cache=False).logits[:, pos_list, :].float()
    ref_nll, ref_ppl = mean_nll_and_ppl(ref_logits, eval_labels)

    out_dir = resolve_output_dir(args, runner_name="runner_inter_optimal_hh")
    save_path = os.path.join(out_dir, "runner_inter_optimal_hh_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.eval_start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "method": "inter_optimal_hh",
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
        print(f"Loaded existing runner_inter_optimal_hh summary: {save_path}")

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
        print(f"\n[runner_inter_optimal_hh] budget={budget}")

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

        inter_optimal_hh_patches, inter_optimal_hh_meta, inter_optimal_hh_losses = build_inter_optimal_hh_patches(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )

        inter_optimal_hh_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_optimal_hh_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_optimal_hh_nll, inter_optimal_hh_ppl = mean_nll_and_ppl(
            inter_optimal_hh_logits,
            eval_labels,
        )

        delta_vs_ref = {
            "inter_optimal_hh_nll_gap": inter_optimal_hh_nll - ref_nll,
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
            "inter_optimal_hh": {"nll": inter_optimal_hh_nll, "ppl": inter_optimal_hh_ppl},
            "delta_vs_ref": delta_vs_ref,
            "losses": {
                "optimal": "loaded_from_existing_summary" if optimal_loaded is not None else "not_available",
                "inter_optimal_hh": inter_optimal_hh_losses,
            },
            "meta": inter_optimal_hh_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = (
            f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, "
            f"inter_optimal_hh={inter_optimal_hh_ppl:.6f}"
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
