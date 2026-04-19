"""
Run baseline and optimal routing online, and compare with q-linear routing.

Inter-Q-Linear routing pipeline per layer:
1) build baseline alpha from QK logits + mask
2) fit per-head linear map W with v_l2_gt objective
3) build alpha_linear from qWk
4) patch hidden states with alpha_linear and run multilayer evaluation

Note:
- This runner only supports tau-target = v_l2_gt.
"""

import os

import torch

from .attention import (
    gen_mask,
)
from .compare_q_linear import build_q_linear_alpha, optimize_w_v_l2

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
    add_tau_target_arg,
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
            "Run baseline/optimal/inter_q_linear routing online and compare PPL. "
            "Inter-Q-Linear routing uses per-head linear map W with v_l2_gt objective."
        ),
        strategy_choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
        default_strategy="h2o",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "Use a different value to test W generalization across samples."
        ),
    )
    parser.add_argument("--w-steps", type=int, default=600)
    parser.add_argument("--w-lr", type=float, default=1e-3)
    parser.add_argument("--w-l2", type=float, default=0.0)
    parser.add_argument(
        "--w-structure",
        type=str,
        default="full",
        choices=["full", "diag"],
        help="Constraint for per-head W in qWk. full: unconstrained matrix; diag: diagonal-only.",
    )

    add_tau_target_arg(parser)

    return finalize_runner_args(parser.parse_args())




def fit_w_by_layer(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs):
    patches_fit = {}
    w_by_layer = {}
    losses_by_layer = {}

    for layer_idx in layer_idx_list:
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches_fit,
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

        w, alpha_linear, w_history, _v_head, _v_gt = optimize_w_v_l2(
            layer_ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            route_mask=route_mask,
            w_steps=args.w_steps,
            w_lr=args.w_lr,
            w_l2=args.w_l2,
            device=ctx.device,
            w_structure=args.w_structure,
        )

        # Keep fit-time online behavior: previous fitted layers are patched when fitting later layers.
        patches_fit[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_linear,
            device=ctx.device,
        )

        w_by_layer[layer_idx] = {
            "w": w,
            "w_history": w_history,
        }
        losses_by_layer[layer_idx] = "skipped_optimal_routing"

        h = w.shape[0]
        d = w.shape[-1]
        eye = torch.eye(d, dtype=w.dtype).unsqueeze(0).expand(h, -1, -1)
        w_delta_norm = float(torch.norm((w - eye).reshape(h, -1), dim=-1).mean().item())
        print(f"[fit W] layer={layer_idx} mean(||W-I||)={w_delta_norm:.6f}")

    return w_by_layer, losses_by_layer


def build_layer_patches_from_w(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs, w_by_layer):
    patches_eval = {}
    for layer_idx in layer_idx_list:
        if layer_idx not in w_by_layer:
            raise KeyError(f"Missing fitted W for layer={layer_idx}")

        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches_eval,
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

        w = w_by_layer[layer_idx]["w"]
        alpha_linear = build_q_linear_alpha(
            layer_ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            w=w,
            route_mask=route_mask,
            device=ctx.device,
        )

        patches_eval[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_linear,
            device=ctx.device,
        )

    return patches_eval


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

    fit_model_inputs = move_model_inputs_to_device(fit_ctx.inputs, fit_ctx.device)
    eval_model_inputs = move_model_inputs_to_device(eval_ctx.inputs, eval_ctx.device)
    eval_labels = get_tail_labels(eval_ctx, pos_list, eval_ctx.device)

    with torch.no_grad():
        ref_logits = eval_ctx.model(**eval_model_inputs, use_cache=False).logits[:, pos_list, :].float()
    ref_nll, ref_ppl = mean_nll_and_ppl(ref_logits, eval_labels)

    out_dir = resolve_output_dir(args, runner_name="runner_inter_q_linear")
    save_path = os.path.join(out_dir, "runner_inter_q_linear_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "tau_target": args.tau_target,
        "w": {
            "w_steps": int(args.w_steps),
            "w_lr": float(args.w_lr),
            "w_l2": float(args.w_l2),
            "w_structure": args.w_structure,
        },
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
                "tau_target",
                "w",
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing runner_inter_q_linear summary: {save_path}")

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
        print(f"\n[runner_inter_q_linear] budget={budget}")

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

        inter_q_linear_meta, inter_q_linear_losses = fit_w_by_layer(
            ctx=fit_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=fit_model_inputs,
        )

        inter_q_linear_patches = build_layer_patches_from_w(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
            w_by_layer=inter_q_linear_meta,
        )

        inter_q_linear_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_q_linear_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_q_linear_nll, inter_q_linear_ppl = mean_nll_and_ppl(inter_q_linear_logits, eval_labels)

        delta_vs_ref = {
            "inter_q_linear_nll_gap": inter_q_linear_nll - ref_nll,
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
            "inter_q_linear": {"nll": inter_q_linear_nll, "ppl": inter_q_linear_ppl},
            "delta_vs_ref": delta_vs_ref,
            "losses": {
                "optimal": "loaded_from_existing_summary" if optimal_loaded is not None else "not_available",
                "inter_q_linear": inter_q_linear_losses,
            },
            "w": inter_q_linear_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, inter_q_linear(v_l2_gt)={inter_q_linear_ppl:.6f}"
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
