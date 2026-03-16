from pathlib import Path

import torch
from torch.nn import functional as F

from .attention import build_qk_routing_alpha, gen_mask
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .sanity import build_modified_attn_hidden, get_tail_labels, unpack_result_entry


def normalize_budget_key(result_dict, target_budget, atol=1e-12):
    for key in result_dict.keys():
        if abs(float(key) - float(target_budget)) <= atol:
            return key
    return None


def compute_metrics(ref_tail_logits, student_tail_logits, labels):
    p_teacher = F.softmax(ref_tail_logits, dim=-1)
    logp_teacher = F.log_softmax(ref_tail_logits, dim=-1)
    logp_student = F.log_softmax(student_tail_logits, dim=-1)
    kl = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1).mean().item()

    teacher_nll = F.cross_entropy(
        ref_tail_logits.reshape(-1, ref_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    ).item()
    student_nll = F.cross_entropy(
        student_tail_logits.reshape(-1, student_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    ).item()

    return {
        "sanity_kl": kl,
        "teacher_nll": teacher_nll,
        "student_nll": student_nll,
        "nll_gap": student_nll - teacher_nll,
    }


def default_output_path(dataset, strategy, loss_type, layers):
    llama_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = llama_dir / "result" / "multilayer" / dataset / strategy / loss_type
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = "-".join(str(x) for x in layers)
    return out_dir / f"layers_{layer_tag}_baseline_compare.pt"


def run_multilayer_baseline_check(
    ctx,
    args,
    target_layers,
    head_idx,
    pos_list,
    model_inputs,
    ref_tail_logits,
):
    llama_dir = Path(__file__).resolve().parent.parent.parent
    result_root = llama_dir / "result"

    labels = get_tail_labels(ctx, pos_list, ctx.device)

    layer_results = {}
    for layer_idx in target_layers:
        result_path = (
            result_root
            / f"layer{layer_idx}"
            / args.dataset
            / args.strategy
            / args.loss_type
            / "result.pt"
        )
        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing result file for layer {layer_idx}: {result_path}. "
                "Please run optimization first."
            )
        layer_results[layer_idx] = torch.load(result_path, weights_only=False)

    summary = {
        "layers": target_layers,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "budgets": {},
    }

    for budget in args.budgets:
        optimal_layer_patch = {}
        baseline_layer_patch = {}

        try:
            for layer_idx in target_layers:
                result_dict = layer_results[layer_idx]
                budget_key = normalize_budget_key(result_dict, budget)
                if budget_key is None:
                    raise KeyError(
                        f"Budget {budget} not found in layer {layer_idx}. "
                        f"Existing budgets: {list(result_dict.keys())}"
                    )

                alpha_opt, _ = unpack_result_entry(result_dict[budget_key])
                # Recompute this layer runtime tensors after applying previous optimal patches.
                optimal_artifacts = capture_layer_artifacts(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    model_inputs=model_inputs,
                    layer_to_patch=optimal_layer_patch,
                )
                optimal_layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, optimal_artifacts)
                optimal_layer_patch[layer_idx] = build_modified_attn_hidden(
                    ctx=optimal_layer_ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=alpha_opt,
                    device=ctx.device,
                )

                # Baseline also needs online layer states under its own previous patches.
                baseline_artifacts = capture_layer_artifacts(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    model_inputs=model_inputs,
                    layer_to_patch=baseline_layer_patch,
                )
                baseline_layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, baseline_artifacts)
                mask = gen_mask(
                    ctx=baseline_layer_ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    head_idx=head_idx,
                    strategy=args.strategy,
                    budget=budget,
                    prompt_len=args.prompt_len,
                    seq_len=args.seq_len,
                )
                alpha_baseline = build_qk_routing_alpha(
                    ctx=baseline_layer_ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    mask=mask,
                    device=ctx.device,
                )
                baseline_layer_patch[layer_idx] = build_modified_attn_hidden(
                    ctx=baseline_layer_ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=alpha_baseline,
                    device=ctx.device,
                )
        except (KeyError, ValueError) as exc:
            print(f"[WARN] Skip budget {budget}: {exc}")
            continue

        optimal_tail_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=optimal_layer_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        baseline_tail_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=baseline_layer_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

        optimal_metrics = compute_metrics(ref_tail_logits, optimal_tail_logits, labels)
        baseline_metrics = compute_metrics(ref_tail_logits, baseline_tail_logits, labels)

        summary["budgets"][float(budget)] = {
            "optimal_sanity_kl": optimal_metrics["sanity_kl"],
            "optimal_teacher_nll": optimal_metrics["teacher_nll"],
            "optimal_student_nll": optimal_metrics["student_nll"],
            "optimal_nll_gap": optimal_metrics["nll_gap"],
            "baseline_sanity_kl": baseline_metrics["sanity_kl"],
            "baseline_teacher_nll": baseline_metrics["teacher_nll"],
            "baseline_student_nll": baseline_metrics["student_nll"],
            "baseline_nll_gap": baseline_metrics["nll_gap"],
            "delta_sanity_kl": baseline_metrics["sanity_kl"] - optimal_metrics["sanity_kl"],
            "delta_student_nll": baseline_metrics["student_nll"] - optimal_metrics["student_nll"],
            "delta_nll_gap": baseline_metrics["nll_gap"] - optimal_metrics["nll_gap"],
        }

        print(
            f"[baseline multi-layer] layers={target_layers}, budget={budget}: "
            f"optimal_KL={optimal_metrics['sanity_kl']:.6f}, "
            f"baseline_KL={baseline_metrics['sanity_kl']:.6f}, "
            f"dKL={summary['budgets'][float(budget)]['delta_sanity_kl']:.6f}, "
            f"dStudentNLL={summary['budgets'][float(budget)]['delta_student_nll']:.6f}, "
            f"dNLLGap={summary['budgets'][float(budget)]['delta_nll_gap']:.6f}"
        )

    out_path = default_output_path(args.dataset, args.strategy, args.loss_type, target_layers)
    torch.save(summary, out_path)
    print(f"Saved multi-layer baseline comparison to: {out_path}")

    return summary, out_path
