import os
import time

import torch

from ..attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong, decompose_count_all_error, get_qk_logits
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


def build_error_runner_parser(description, method_name):
    parser = create_base_runner_parser(
        description=description,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        strategy_help=f"{method_name} correction requires h2o_with_belong_all metadata.",
        eval_start_help=(
            "Start index for evaluation sample. If not set, use --start. "
            f"For {method_name} this only changes evaluation sample; no fitted params are reused."
        ),
    )
    parser.add_argument(
        "--merge-metric",
        choices=["k", "v"],
        default="k",
        help="Cluster merge target metric. Default is k; v is allowed only for diagnostics.",
    )
    return finalize_runner_args(parser.parse_args())


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
    return attn_hidden.detach()


def _build_error_corrected_patches(
    ctx,
    args,
    budget,
    layer_idx_list,
    head_idx,
    pos_list,
    model_inputs,
    correction_name,
):
    if correction_name not in {"count_all", "key", "value", "kv"}:
        raise ValueError(f"Unsupported correction_name={correction_name}")

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

        route_mask, belong, count = gen_mask_h2o_with_belong_all(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            budget=budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
            merge_metric=args.merge_metric,
        )
        belong_root = canonicalize_belong(belong, pos_list)

        qk_logits = get_qk_logits(
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

        decomp = decompose_count_all_error(
            qk_logits=qk_logits,
            v_head=v_head,
            route_mask=route_mask,
            belong_root=belong_root,
            count=count,
            pos_list=pos_list,
        )

        if correction_name == "count_all":
            v_new = decomp["count_all_v"]
        elif correction_name == "key":
            v_new = decomp["key_corrected_v"]
        elif correction_name == "value":
            v_new = decomp["value_corrected_v"]
        else:
            v_new = decomp["reconstructed_v"]

        patches[layer_idx] = build_modified_attn_hidden_from_vnew(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            v_new=v_new,
            device=ctx.device,
        )

        count_l2 = torch.norm(decomp["count_all_v"] - v_gt, p=2, dim=-1).mean()
        corrected_l2 = torch.norm(v_new - v_gt, p=2, dim=-1).mean()
        recon_l2 = torch.norm(decomp["reconstructed_v"] - v_gt, p=2, dim=-1).mean()
        key_term_l2 = torch.norm(decomp["key_term_v"], p=2, dim=-1).mean()
        value_term_l2 = torch.norm(decomp["value_term_v"], p=2, dim=-1).mean()
        den_rel_gap = (
            decomp["key_den_error"].abs() / decomp["full_den"].abs().clamp_min(1e-30)
        ).mean()

        layer_elapsed = time.time() - layer_start
        total_elapsed = time.time() - build_start
        avg_layer_time = total_elapsed / float(layer_ord)
        eta_seconds = avg_layer_time * (total_layers - layer_ord)

        print(
            f"[inter_{correction_name}_error][budget={budget}] layer {layer_idx} "
            f"({layer_ord}/{total_layers}) done | "
            f"layer_time={layer_elapsed:.2f}s elapsed={total_elapsed:.2f}s "
            f"eta={eta_seconds:.2f}s"
        )

        layer_meta[layer_idx] = {
            "correction": correction_name,
            "merge_metric": args.merge_metric,
            "mean_count_all_v_l2": float(count_l2.item()),
            "mean_corrected_v_l2": float(corrected_l2.item()),
            "mean_reconstructed_v_l2": float(recon_l2.item()),
            "mean_key_term_l2": float(key_term_l2.item()),
            "mean_value_term_l2": float(value_term_l2.item()),
            "mean_den_rel_gap": float(den_rel_gap.item()),
            "layer_time_sec": float(layer_elapsed),
            "eta_after_layer_sec": float(eta_seconds),
        }

    return patches, layer_meta


def run_error_correction_runner(args, correction_name, runner_name, summary_filename):
    set_seed(42)
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

    out_dir = resolve_output_dir(args, runner_name=runner_name)
    save_path = os.path.join(out_dir, summary_filename)

    summary = {
        "dataset": args.dataset,
        "fit_start": args.start,
        "eval_start": args.eval_start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "method": correction_name,
        "merge_metric": args.merge_metric,
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
                "merge_metric",
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing {runner_name} summary: {save_path}")

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
        print(f"\n[{runner_name}] budget={budget}")

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

        corrected_patches, corrected_meta = _build_error_corrected_patches(
            ctx=eval_ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
            correction_name=correction_name,
        )

        corrected_logits = run_with_multilayer_patches(
            ctx=eval_ctx,
            layer_to_patch=corrected_patches,
            pos_list=pos_list,
            model_inputs=eval_model_inputs,
        )
        corrected_nll, corrected_ppl = mean_nll_and_ppl(corrected_logits, eval_labels)

        corrected_key = "count_all" if correction_name == "count_all" else f"inter_{correction_name}_error"
        delta_vs_ref = {
            f"{corrected_key}_nll_gap": corrected_nll - ref_nll,
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
            corrected_key: {"nll": corrected_nll, "ppl": corrected_ppl},
            "delta_vs_ref": delta_vs_ref,
            "meta": corrected_meta,
            "loaded_metrics": loaded_metrics,
        }

        msg = (
            f"[ppl@eval_start={args.eval_start}] ref={ref_ppl:.6f}, "
            f"{corrected_key}={corrected_ppl:.6f}"
        )
        if baseline_ppl is not None:
            msg += f", baseline={baseline_ppl:.6f}"
        if optimal_ppl is not None:
            msg += f", optimal={optimal_ppl:.6f}"
        print(msg)

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")
