"""
Run baseline and optimal routing online, and compare with q-bias routing.

Inter-Q-Bias routing pipeline per layer:
1) build baseline alpha from QK logits + mask
2) fit per-head per-key bias with v_l2_gt objective
3) build alpha_bias from qk + bias
4) patch hidden states with alpha_bias and run multilayer evaluation

Note:
- This runner only supports tau-target = v_l2_gt.
"""

import os

import torch

from .attention import gen_mask, get_attention_map_after_rope
from .compare_q_bias import build_bias_routing_alpha, optimize_bias_v_l2
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
            "Run baseline/optimal/inter_q_bias routing online and compare PPL. "
            "Inter-Q-Bias routing uses per-head per-key bias with v_l2_gt objective."
        ),
        strategy_choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
        default_strategy="h2o",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            "Use a different value to test bias generalization across samples."
        ),
    )
    parser.add_argument("--bias-steps", type=int, default=500)
    parser.add_argument("--bias-lr", type=float, default=5e-2)
    parser.add_argument("--bias-l2", type=float, default=0.0)

    add_tau_target_arg(parser)

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


def fit_bias_by_layer(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs):
    patches_fit = {}
    bias_by_layer = {}
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

        mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
            budget=budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )

        # _alpha_base = build_qk_routing_alpha(
        #     ctx=layer_ctx,
        #     layer_idx=layer_idx,
        #     head_idx=head_idx,
        #     pos_list=pos_list,
        #     mask=mask,
        #     device=ctx.device,
        # )

        qk_logits = _get_qk_logits(
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

        bias, alpha_bias, bias_history = optimize_bias_v_l2(
            v_head=v_head,
            v_gt=v_gt,
            qk_logits=qk_logits,
            mask=mask,
            bias_steps=args.bias_steps,
            bias_lr=args.bias_lr,
            bias_l2=args.bias_l2,
        )

        alpha_bias = build_bias_routing_alpha(qk_logits=qk_logits, mask=mask, bias=bias)

        # Keep fit-time online behavior: previous fitted layers are patched when fitting later layers.
        patches_fit[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_bias,
            device=ctx.device,
        )

        bias_by_layer[layer_idx] = {
            "bias": bias,
            "bias_history": bias_history,
        }
        losses_by_layer[layer_idx] = "skipped_optimal_routing"

        bias_abs_mean = float(bias.abs().mean().item())
        print(f"[fit bias] layer={layer_idx} mean(|bias|)={bias_abs_mean:.6f}")

    return bias_by_layer, losses_by_layer


def build_layer_patches_from_bias(
    ctx,
    args,
    budget,
    layer_idx_list,
    head_idx,
    pos_list,
    model_inputs,
    bias_by_layer,
):
    patches_eval = {}
    for layer_idx in layer_idx_list:
        if layer_idx not in bias_by_layer:
            raise KeyError(f"Missing fitted bias for layer={layer_idx}")

        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches_eval,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

        mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
            budget=budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )

        qk_logits = _get_qk_logits(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            device=ctx.device,
        )

        bias = bias_by_layer[layer_idx]["bias"]
        alpha_bias = build_bias_routing_alpha(qk_logits=qk_logits, mask=mask, bias=bias)

        patches_eval[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_bias,
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

    out_dir = resolve_output_dir(args, runner_name="runner_inter_q_bias")
    save_path = os.path.join(out_dir, "runner_inter_q_bias_summary.pt")

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "tau_target": args.tau_target,
        "bias": {
            "bias_steps": int(args.bias_steps),
            "bias_lr": float(args.bias_lr),
            "bias_l2": float(args.bias_l2),
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
                "bias",
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing runner_inter_q_bias summary: {save_path}")

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
        print(f"\n[runner_inter_q_bias] budget={budget}")

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

        inter_q_bias_meta, inter_q_bias_losses = fit_bias_by_layer(
            ctx=fit_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=fit_model_inputs,
        )

        inter_q_bias_patches = build_layer_patches_from_bias(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
            bias_by_layer=inter_q_bias_meta,
        )

        inter_q_bias_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=inter_q_bias_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        inter_q_bias_nll, inter_q_bias_ppl = mean_nll_and_ppl(inter_q_bias_logits, eval_labels)

        delta_vs_ref = {
            "inter_q_bias_nll_gap": inter_q_bias_nll - ref_nll,
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
            "inter_q_bias": {"nll": inter_q_bias_nll, "ppl": inter_q_bias_ppl},
            "delta_vs_ref": delta_vs_ref,
            "losses": {
                "optimal": "loaded_from_existing_summary" if optimal_loaded is not None else "not_available",
                "inter_q_bias": inter_q_bias_losses,
            },
            "bias": inter_q_bias_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, inter_q_bias(v_l2_gt)={inter_q_bias_ppl:.6f}"
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
