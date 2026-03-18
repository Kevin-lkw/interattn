from pathlib import Path

import torch
from torch.nn import functional as F

from .attention import build_qk_routing_alpha, gen_mask
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
    # if result already exists, skip computation and directly load for summary printing
    out_path  = Path(f"../result/{args.dataset}/{args.strategy}/qk_routing.pt")
    
    if out_path.exists():
        print(f"Found existing baseline comparison result at {out_path}, loading...")
        summary = torch.load(out_path)
        print("Loaded summary:")
        print(summary)
        return summary, out_path
    labels = get_tail_labels(ctx, pos_list, ctx.device)

    summary = {
        "layers": target_layers,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "budgets": {},
    }

    for budget in args.budgets:
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
                mask = gen_mask(
                    ctx=baseline_layer_ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    head_idx=head_idx,
                    strategy=args.strategy,
                    budget=budget,
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

        summary["budgets"][float(budget)] = baseline_metrics

    torch.save(summary, out_path)
    print(f"Saved multi-layer baseline comparison to: {out_path}")

    return summary, out_path
