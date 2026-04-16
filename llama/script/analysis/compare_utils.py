import os

import matplotlib.pyplot as plt
import torch

from .attention import build_qk_routing_alpha
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import get_result_path, normalize_budget_key
from .sanity import build_modified_attn_hidden


def validate_common_args(args, num_layers, num_heads):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.head is not None and (args.head < 0 or args.head >= num_heads):
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.heads is not None:
        for h in args.heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Invalid --heads entry {h}; expected [0, {num_heads - 1}]")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")


def resolve_head_indices(args, num_heads):
    if args.heads is not None and len(args.heads) > 0:
        return sorted(set(int(x) for x in args.heads))
    if args.head is not None:
        return [int(args.head)]
    return list(range(num_heads))


def resolve_output_dir(args, head_idx, compare_tag, include_loss_type=True):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        if len(head_idx) == 1:
            head_tag = f"head{head_idx[0]}"
        else:
            head_tag = f"heads_{len(head_idx)}"

        if include_loss_type:
            out_dir = (
                f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
                f"{args.loss_type}/{compare_tag}/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
            )
        else:
            out_dir = (
                f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
                f"{compare_tag}/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
            )

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_saved_patch_hidden_for_layer(args, layer_idx, budget, device, loss_type_override=None):
    loss_type = args.loss_type if loss_type_override is None else loss_type_override
    path = get_result_path(
        layer_idx=layer_idx,
        dataset=args.dataset,
        start=args.start,
        adaptive_budget=args.adaptive_budget,
        strategy=args.strategy,
        loss_type=loss_type,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing optimal layer result for layer={layer_idx}. Expected file: {path}"
        )

    result = torch.load(path, map_location="cpu", weights_only=False)
    key = normalize_budget_key(result, budget)
    if key is None:
        raise KeyError(
            f"Budget {budget} not found in layer={layer_idx} result keys: {list(result.keys())}"
        )

    entry = result[key]
    if not isinstance(entry, dict) or "patch_hidden" not in entry:
        raise KeyError(
            f"layer={layer_idx}, budget={budget} has no patch_hidden in saved result entry."
        )
    return entry["patch_hidden"].to(device)


def build_optimal_saved_prefix_patches(args, target_layer, budget, device, loss_type_override=None):
    patches = {}
    for layer_idx in range(target_layer):
        patches[layer_idx] = load_saved_patch_hidden_for_layer(
            args=args,
            layer_idx=layer_idx,
            budget=budget,
            device=device,
            loss_type_override=loss_type_override,
        )
    return patches


def build_baseline_prefix_patches(ctx, args, target_layer, pos_list, model_inputs, build_mask_fn):
    patches = {}
    head_idx = list(range(ctx.model_config.num_attention_heads))
    for layer_idx in range(target_layer):
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

        mask = build_mask_fn(layer_ctx, layer_idx, head_idx)

        alpha_baseline = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=mask,
            device=ctx.device,
        )
        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_baseline,
            device=ctx.device,
        )
        print(f"[prefix baseline rebuild] layer {layer_idx} done")
    return patches


def save_per_pos_metric_tsv(out_path, pos_list, base_metric, other_metric, other_name):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"pos\tbase_v_l2\t{other_name}_v_l2\tdelta_base_minus_{other_name}\n")
        for i, pos in enumerate(pos_list):
            mb = float(base_metric[i].item())
            mo = float(other_metric[i].item())
            f.write(f"{pos}\t{mb:.8e}\t{mo:.8e}\t{(mb - mo):.8e}\n")


def plot_per_pos_two_lines(
    out_path,
    pos_list,
    y1,
    y2,
    label1,
    label2,
    title,
    dpi=180,
    xlabel="pos",
    ylabel="v_l2",
):
    x = pos_list
    yy1 = y1.detach().float().cpu().tolist()
    yy2 = y2.detach().float().cpu().tolist()

    plt.figure(figsize=(10, 4.8))
    plt.plot(x, yy1, linewidth=1.4, alpha=0.9, label=label1)
    plt.plot(x, yy2, linewidth=1.4, alpha=0.9, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
