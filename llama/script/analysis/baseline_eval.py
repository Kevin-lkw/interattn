from pathlib import Path
from sqlite3 import adapt

import torch
from torch.nn import functional as F

from .attention import (
    build_kept_kv_cache,
    build_modified_attn_hidden_from_kept_v,
    build_qk_routing_alpha,
    build_qk_routing_alpha_on_kept_kv,
    gen_mask,
)
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .sanity import build_modified_attn_hidden, get_tail_labels, compute_metrics




def run_multilayer_baseline_check(
    ctx,
    args,
    target_layers,
    head_idx,
    pos_list,
    model_inputs,
    ref_tail_logits,
):
    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    if args.kv_compress_mode == "mask":
        out_path = Path(f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/qk_routing.pt")
    elif args.kv_compress_mode == "kept_kv":
        out_path = Path(f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/qk_routing_kept_kv.pt")
    else:
        raise ValueError(f"Unknown kv_compress_mode: {args.kv_compress_mode}")

    if out_path.exists():
        print(f"Found existing baseline comparison result at {out_path}, loading...")
        summary = torch.load(out_path)
    else:
        summary = {
            "layers": target_layers,
            "dataset": args.dataset,
            "start": args.start,
            "strategy": args.strategy,
            "compress_mode": args.kv_compress_mode,
            "budgets": {},
        }

    existing_budget_keys = summary.get("budgets", {}).keys()
    existing_budgets = set()
    for key in existing_budget_keys:
        try:
            existing_budgets.add(float(key))
        except (TypeError, ValueError):
            continue

    missing_budgets = [budget for budget in args.budgets if float(budget) not in existing_budgets]
    if not missing_budgets:
        print("No new budgets to evaluate. Reusing existing baseline summary.")
        print(summary)
        return summary, out_path

    labels = get_tail_labels(ctx, pos_list, ctx.device)

    for budget in missing_budgets:
        budget_key = float(budget)
        baseline_layer_patch = {}

        try:
            for layer_idx in target_layers:
                # Baseline needs online layer states under its own previous patches.
                baseline_artifacts = capture_layer_artifacts(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    model_inputs=model_inputs,
                    layer_to_patch=baseline_layer_patch,
                )
                baseline_layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, baseline_artifacts)
                if args.kv_compress_mode == "mask":
                    mask = gen_mask(
                        ctx=baseline_layer_ctx,
                        layer_idx=layer_idx,
                        pos_list=pos_list,
                        head_idx=head_idx,
                        strategy=args.strategy,
                        budget=budget,
                        seq_len=args.seq_len,
                        adaptive_budget=args.adaptive_budget,
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
                elif args.kv_compress_mode == "kept_kv":
                    kept_cache = build_kept_kv_cache(
                        ctx=baseline_layer_ctx,
                        layer_idx=layer_idx,
                        pos_list=pos_list,
                        head_idx=head_idx,
                        strategy=args.strategy,
                        budget=budget,
                        seq_len=args.seq_len,
                        adaptive_budget=args.adaptive_budget,
                        device=ctx.device,
                    )
                    alpha_baseline = build_qk_routing_alpha_on_kept_kv(
                        ctx=baseline_layer_ctx,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        pos_list=pos_list,
                        kept_k=kept_cache["kept_k"],
                        keep_valid=kept_cache["keep_valid"],
                        device=ctx.device,
                    )
                    baseline_layer_patch[layer_idx] = build_modified_attn_hidden_from_kept_v(
                        ctx=baseline_layer_ctx,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        pos_list=pos_list,
                        alpha=alpha_baseline,
                        kept_v=kept_cache["kept_v"],
                        device=ctx.device,
                    )
                else:
                    raise ValueError(f"Unknown kv_compress_mode: {args.kv_compress_mode}")
        except ValueError as exc:
            print(f"[WARN] Skip budget {budget}: {exc}")
            continue

        baseline_tail_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=baseline_layer_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

        baseline_metrics = compute_metrics(ref_tail_logits, baseline_tail_logits, labels)

        summary["budgets"][budget_key] = baseline_metrics

    torch.save(summary, out_path)
    print(f"Saved/updated multi-layer baseline comparison to: {out_path}")

    return summary, out_path
