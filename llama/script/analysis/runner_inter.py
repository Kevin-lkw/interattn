import argparse
import os

import torch
from torch.nn import functional as F

from .attention import build_qk_routing_alpha, gen_mask, optimize_alpha_star
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
            "Run baseline/optimal/inter routing online and compare PPL. "
            "Inter routing replaces top-K%% entries ranked by |alpha_opt-alpha_base|*|V|."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--strategy", type=str, default="h2o", choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--loss-type", type=str, default="v_l2", choices=["logits_kl", "v_l2", "v_kl"])

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1])
    parser.add_argument(
        "--replace-k-percent",
        type=float,
        default=50.0,
        help="Replace top K percent entries per (head, query position), ranked by |diff|*|V|.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/runner_inter/kpct_{args.replace_k_percent:g}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _adaptive_str(adaptive_budget):
    return "adaptive" if adaptive_budget else "fixed"


def _nll_to_ppl(nll):
    return float(torch.exp(torch.tensor(float(nll), dtype=torch.float32)).item())


def load_existing_baseline_metrics(args, budget):
    # Produced by baseline_eval.py via runner.py
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
        raise KeyError(
            f"Budget {budget} not found in baseline summary keys: {list(budgets.keys())}"
        )

    entry = budgets[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid baseline entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {
        "nll": nll,
        "ppl": _nll_to_ppl(nll),
        "raw": entry,
    }


def load_existing_optimal_metrics(args, budget):
    # Produced by runner.py save_results(...)
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
        raise KeyError(
            f"Budget {budget} not found in optimal summary keys: {list(summary.keys())}"
        )

    entry = summary[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid optimal entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {
        "nll": nll,
        "ppl": _nll_to_ppl(nll),
        "raw": entry,
    }


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


def _build_alpha_inter(alpha_base, alpha_opt, v_selected, mask, pos_list, replace_k_percent):
    """
    Build mixed routing alpha by replacing top-K% entries of baseline with optimal.

    Ranking score: |alpha_opt - alpha_base| * |V|, evaluated per (head, query row, key).
    """
    if replace_k_percent < 0 or replace_k_percent > 100:
        raise ValueError("--replace-k-percent must be in [0, 100]")

    alpha_inter = alpha_base.clone()
    n_heads = alpha_base.shape[0]

    v_abs = torch.norm(v_selected.detach().float(), p=2, dim=-1)  # [n_heads, seq_len]

    for h in range(n_heads):
        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            valid_mask = torch.isfinite(mask[h, row_i, :total_available])
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            k_valid = int(valid_idx.numel())
            if k_valid <= 0:
                continue

            n_replace = int(round(k_valid * replace_k_percent / 100.0))
            n_replace = max(0, min(k_valid, n_replace))
            if n_replace == 0:
                continue

            diff_abs = (alpha_opt[h, row_i, valid_idx] - alpha_base[h, row_i, valid_idx]).abs()
            score = diff_abs * v_abs[h, valid_idx]
            top_idx_local = torch.topk(score, k=n_replace, largest=True).indices
            replace_idx = valid_idx[top_idx_local]

            alpha_inter[h, row_i, replace_idx] = alpha_opt[h, row_i, replace_idx]

    return alpha_inter


def build_layer_patches_inter(ctx, args, budget, layer_idx_list, head_idx, pos_list, model_inputs):
    patches = {}
    loss_by_layer = {}
    for layer_idx in layer_idx_list:
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
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

        alpha_base = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=mask,
            device=ctx.device,
        )
        alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            training_steps=args.training_steps,
            lr=args.lr,
            mask=mask,
            loss_type=args.loss_type,
            device=ctx.device,
        )

        v_selected = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()
        alpha_inter = _build_alpha_inter(
            alpha_base=alpha_base,
            alpha_opt=alpha_opt,
            v_selected=v_selected,
            mask=mask,
            pos_list=pos_list,
            replace_k_percent=args.replace_k_percent,
        )

        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_inter,
            device=ctx.device,
        )
        loss_by_layer[layer_idx] = losses
    return patches, loss_by_layer


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
    save_path = os.path.join(out_dir, "runner_inter_summary.pt")

    summary = {
        "dataset": args.dataset,
        "start": args.start,
        "seq_len": args.seq_len,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "replace_k_percent": float(args.replace_k_percent),
        "budgets": [float(x) for x in args.budgets],
        "layers": layer_idx_list,
        "reference": {
            "nll": ref_nll,
            "ppl": ref_ppl,
        },
        "results": {},
    }

    if os.path.exists(save_path):
        old_summary = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(old_summary, dict):
            # Preserve previous results; only append newly requested budgets.
            old_results = old_summary.get("results", {})
            if isinstance(old_results, dict):
                summary["results"].update(old_results)
            # Keep previous metadata where possible to avoid accidental overwrite.
            for key in [
                "dataset",
                "start",
                "seq_len",
                "strategy",
                "loss_type",
                "replace_k_percent",
                "layers",
                "reference",
            ]:
                if key in old_summary:
                    summary[key] = old_summary[key]
        print(f"Loaded existing runner_inter summary: {save_path}")

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
        print(f"\n[runner_inter] budget={budget}")

        baseline_loaded = load_existing_baseline_metrics(args=args, budget=budget)
        optimal_loaded = load_existing_optimal_metrics(args=args, budget=budget)
        baseline_nll, baseline_ppl = baseline_loaded["nll"], baseline_loaded["ppl"]
        optimal_nll, optimal_ppl = optimal_loaded["nll"], optimal_loaded["ppl"]
        print(
            f"[loaded metrics] baseline_nll={baseline_nll:.6f}, baseline_ppl={baseline_ppl:.6f}, "
            f"optimal_nll={optimal_nll:.6f}, optimal_ppl={optimal_ppl:.6f}"
        )
        inter_patches, inter_loss = build_layer_patches_inter(
            ctx=ctx,
            args=args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        inter_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=inter_patches,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        inter_nll, inter_ppl = mean_nll_and_ppl(inter_logits, labels)

        summary["results"][float(budget)] = {
            "baseline": {"nll": baseline_nll, "ppl": baseline_ppl},
            "optimal": {"nll": optimal_nll, "ppl": optimal_ppl},
            "inter": {"nll": inter_nll, "ppl": inter_ppl},
            "delta_vs_ref": {
                "baseline_nll_gap": baseline_nll - ref_nll,
                "optimal_nll_gap": optimal_nll - ref_nll,
                "inter_nll_gap": inter_nll - ref_nll,
            },
            "losses": {
                "optimal": "loaded_from_existing_summary",
                "inter": inter_loss,
            },
            "loaded_metrics": {
                "baseline": baseline_loaded["raw"],
                "optimal": optimal_loaded["raw"],
            },
        }

        print(
            f"[ppl] ref={ref_ppl:.6f}, baseline={baseline_ppl:.6f}, "
            f"inter(K={args.replace_k_percent:g}%)={inter_ppl:.6f}, optimal={optimal_ppl:.6f}"
        )

    torch.save(summary, save_path)
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
