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

import argparse
import os

import torch
from torch.nn import functional as F

from .attention import (
    build_qk_routing_alpha,
    gen_mask,
    get_attention_map_after_rope,
    optimize_alpha_star,
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
from .sanity import build_modified_attn_hidden, get_tail_labels, move_model_inputs_to_device

from .compare_q_linear import optimize_w_v_l2


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline/optimal/inter_q_linear routing online and compare PPL. "
            "Inter-Q-Linear routing uses per-head linear map W with v_l2_gt objective."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05])

    parser.add_argument("--w-steps", type=int, default=600)
    parser.add_argument("--w-lr", type=float, default=1e-3)
    parser.add_argument("--w-l2", type=float, default=0.0)

    parser.add_argument(
        "--tau-target",
        type=str,
        default="v_l2_gt",
        choices=["v_l2_gt"],
        help="Fixed target for this runner.",
    )

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    if args.layers is None:
        args.all_layers = True
    else:
        args.all_layers = False
    return args


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/runner_inter_q_linear"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _adaptive_str(adaptive_budget):
    return "adaptive" if adaptive_budget else "fixed"


def _nll_to_ppl(nll):
    return float(torch.exp(torch.tensor(float(nll), dtype=torch.float32)).item())


def load_existing_baseline_metrics(args, budget):
    path = (
        f"../result/{args.dataset}_{args.start}/"
        f"{_adaptive_str(args.adaptive_budget)}/{args.strategy}/qk_routing.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing baseline summary file: {path}. "
            "Please run runner.py baseline comparison first."
        )

    summary = torch.load(path, map_location="cpu", weights_only=False)
    budgets = summary.get("budgets", {}) if isinstance(summary, dict) else {}
    key = normalize_budget_key(budgets, budget)
    if key is None:
        raise KeyError(f"Budget {budget} not found in baseline summary keys: {list(budgets.keys())}")

    entry = budgets[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid baseline entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {"nll": nll, "ppl": _nll_to_ppl(nll), "raw": entry}


def load_existing_optimal_metrics(args, budget):
    path = (
        f"../result/{args.dataset}_{args.start}/"
        f"{_adaptive_str(args.adaptive_budget)}/{args.strategy}/{args.loss_type}/"
        "layer_all/budget_to_final_metrics.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing optimal summary file: {path}. "
            "Please run runner.py optimization first."
        )

    summary = torch.load(path, map_location="cpu", weights_only=False)
    key = normalize_budget_key(summary, budget)
    if key is None:
        raise KeyError(f"Budget {budget} not found in optimal summary keys: {list(summary.keys())}")

    entry = summary[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid optimal entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {"nll": nll, "ppl": _nll_to_ppl(nll), "raw": entry}


def mean_nll_and_ppl(logits, labels):
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )
    ppl = torch.exp(nll)
    return float(nll.item()), float(ppl.item())


def _resolve_head_indices(num_heads):
    return list(range(num_heads))


def build_layer_patches_inter_q_linear(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs):
    patches = {}
    w_by_layer = {}
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
        )

        patches[layer_idx] = build_modified_attn_hidden(
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
        print(f"[inter_q_linear] layer={layer_idx} mean(||W-I||)={w_delta_norm:.6f}")

    return patches, losses_by_layer, w_by_layer


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    ctx = load_context(args, dtype=dtype, device=args.device)
    validate_args_with_cache(ctx, args)
    ctx.model.eval()

    head_idx = _resolve_head_indices(ctx.model_config.num_attention_heads)
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)

    with torch.no_grad():
        ref_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    ref_nll, ref_ppl = mean_nll_and_ppl(ref_logits, labels)

    out_dir = resolve_output_dir(args)
    save_path = os.path.join(out_dir, "runner_inter_q_linear_summary.pt")

    summary = {
        "dataset": args.dataset,
        "start": args.start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "tau_target": args.tau_target,
        "w": {
            "w_steps": int(args.w_steps),
            "w_lr": float(args.w_lr),
            "w_l2": float(args.w_l2),
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
                "start",
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

        baseline_loaded = load_existing_baseline_metrics(args=args, budget=budget)
        optimal_loaded = load_existing_optimal_metrics(args=args, budget=budget)
        baseline_nll, baseline_ppl = baseline_loaded["nll"], baseline_loaded["ppl"]
        optimal_nll, optimal_ppl = optimal_loaded["nll"], optimal_loaded["ppl"]
        print(
            f"[loaded metrics] baseline_nll={baseline_nll:.6f}, baseline_ppl={baseline_ppl:.6f}, "
            f"optimal_nll={optimal_nll:.6f}, optimal_ppl={optimal_ppl:.6f}"
        )

        inter_q_linear_patches, inter_q_linear_losses, inter_q_linear_meta = build_layer_patches_inter_q_linear(
            ctx=ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

        inter_q_linear_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=inter_q_linear_patches,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        inter_q_linear_nll, inter_q_linear_ppl = mean_nll_and_ppl(inter_q_linear_logits, labels)

        summary["results"][float(budget)] = {
            "baseline": {"nll": baseline_nll, "ppl": baseline_ppl},
            "optimal": {"nll": optimal_nll, "ppl": optimal_ppl},
            "inter_q_linear": {"nll": inter_q_linear_nll, "ppl": inter_q_linear_ppl},
            "delta_vs_ref": {
                "baseline_nll_gap": baseline_nll - ref_nll,
                "optimal_nll_gap": optimal_nll - ref_nll,
                "inter_q_linear_nll_gap": inter_q_linear_nll - ref_nll,
            },
            "losses": {
                "optimal": "loaded_from_existing_summary",
                "inter_q_linear": inter_q_linear_losses,
            },
            "w": inter_q_linear_meta,
            "loaded_metrics": {
                "baseline": baseline_loaded["raw"],
                "optimal": optimal_loaded["raw"],
            },
        }

        print(
            f"[ppl] ref={ref_ppl:.6f}, baseline={baseline_ppl:.6f}, "
            f"inter_q_linear(v_l2_gt)={inter_q_linear_ppl:.6f}, optimal={optimal_ppl:.6f}"
        )

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
