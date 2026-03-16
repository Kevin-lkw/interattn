import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

try:
    from .runner import load_context, validate_args_with_cache
    from .config import set_seed, str_to_torch_dtype
    from .sanity import (
        build_modified_attn_hidden,
        get_tail_labels,
        move_model_inputs_to_device,
        unpack_result_entry,
    )
except ImportError:
    from runner import load_context, validate_args_with_cache
    from config import set_seed, str_to_torch_dtype
    from sanity import (
        build_modified_attn_hidden,
        get_tail_labels,
        move_model_inputs_to_device,
        unpack_result_entry,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate final performance by replacing multiple layers with learned alpha* simultaneously"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model-name", type=str, default="llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--strategy", type=str, default="h2o")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
        help="Loss type subdirectory to read from",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--tail-len", type=int, default=64)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to replace simultaneously",
    )
    layer_group.add_argument(
        "--all-layers",
        action="store_true",
        help="Process all transformer layers",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
        help="Budgets to evaluate. Each budget must exist in every selected layer result.pt",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional output .pt path. If empty, save to llama/result/multilayer/<dataset>/<strategy>/",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot metric curves against budget after evaluation",
    )
    parser.add_argument(
        "--plot-metrics",
        type=str,
        nargs="+",
        default=["sanity_kl", "nll_gap"],
        choices=["sanity_kl", "teacher_nll", "student_nll", "nll_gap"],
        help="Metrics to plot on y-axis",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="Optional output image path; if empty, save near output .pt",
    )
    parser.add_argument(
        "--plot-logx",
        action="store_true",
        help="Use log scale for budget axis",
    )
    parser.add_argument(
        "--plot-logy",
        action="store_true",
        help="Use log scale for metric axis",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot window",
    )
    parser.add_argument(
        "--error-bar",
        type=str,
        default="std",
        choices=["none", "std", "sem"],
        help="Error bar type computed over tail tokens for each budget",
    )
    return parser.parse_args()


def normalize_budget_key(result_dict, target_budget, atol=1e-12):
    for key in result_dict.keys():
        if abs(float(key) - float(target_budget)) <= atol:
            return key
    return None


def get_alpha_for_layer_budget(result_root, layer_idx, dataset, strategy, loss_type, budget):
    result_path = result_root / f"layer{layer_idx}" / dataset / strategy / loss_type / "result.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file for layer {layer_idx}: {result_path}")

    result = torch.load(result_path, weights_only=False)
    budget_key = normalize_budget_key(result, budget)
    if budget_key is None:
        raise KeyError(
            f"Budget {budget} not found in layer {layer_idx} result. Existing: {list(result.keys())}"
        )

    alpha, _ = unpack_result_entry(result[budget_key])
    return alpha


def run_with_multilayer_patches(ctx, layer_to_patch, pos_list, model_inputs):
    if not layer_to_patch:
        raise ValueError("layer_to_patch is empty")

    pos_idx = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    handles = []

    def hook_factory(layer_idx):
        patch_hidden = layer_to_patch[layer_idx]

        def _hook(_module, _module_inputs, module_output):
            if isinstance(module_output, tuple):
                attn_out = module_output[0].clone()
                attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
                return (attn_out,) + module_output[1:]

            attn_out = module_output.clone()
            attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
            return attn_out

        return _hook

    for layer_idx in layer_to_patch.keys():
        layer = ctx.model.model.layers[layer_idx]
        handle = layer.self_attn.register_forward_hook(hook_factory(layer_idx))
        handles.append(handle)

    try:
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    finally:
        for h in handles:
            h.remove()

    return logits


def compute_metrics(ref_tail_logits, student_tail_logits, labels):
    p_teacher = F.softmax(ref_tail_logits, dim=-1)
    logp_teacher = F.log_softmax(ref_tail_logits, dim=-1)
    logp_student = F.log_softmax(student_tail_logits, dim=-1)
    kl_token = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)  # [B, T]
    kl = kl_token.mean().item()

    teacher_nll_token = F.cross_entropy(
        ref_tail_logits.reshape(-1, ref_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(ref_tail_logits.size(0), ref_tail_logits.size(1))
    teacher_nll = teacher_nll_token.mean().item()

    student_nll_token = F.cross_entropy(
        student_tail_logits.reshape(-1, student_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(student_tail_logits.size(0), student_tail_logits.size(1))
    student_nll = student_nll_token.mean().item()

    nll_gap_token = student_nll_token - teacher_nll_token

    # reduce batch dimension first, then compute dispersion across tail tokens
    kl_token_mean = kl_token.mean(dim=0)
    teacher_nll_token_mean = teacher_nll_token.mean(dim=0)
    student_nll_token_mean = student_nll_token.mean(dim=0)
    nll_gap_token_mean = nll_gap_token.mean(dim=0)

    def std_sem(x):
        std = x.std(unbiased=False).item()
        sem = (std / (x.numel() ** 0.5)) if x.numel() > 0 else 0.0
        return std, sem

    kl_std, kl_sem = std_sem(kl_token_mean)
    teacher_nll_std, teacher_nll_sem = std_sem(teacher_nll_token_mean)
    student_nll_std, student_nll_sem = std_sem(student_nll_token_mean)
    nll_gap_std, nll_gap_sem = std_sem(nll_gap_token_mean)

    return {
        "sanity_kl": kl,
        "sanity_kl_std": kl_std,
        "sanity_kl_sem": kl_sem,
        "teacher_nll": teacher_nll,
        "teacher_nll_std": teacher_nll_std,
        "teacher_nll_sem": teacher_nll_sem,
        "student_nll": student_nll,
        "student_nll_std": student_nll_std,
        "student_nll_sem": student_nll_sem,
        "nll_gap": student_nll - teacher_nll,
        "nll_gap_std": nll_gap_std,
        "nll_gap_sem": nll_gap_sem,
    }


def default_output_path(dataset, strategy, loss_type, layers):
    llama_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = llama_dir / "result" / "multilayer" / dataset / strategy / loss_type
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = "-".join(str(x) for x in layers)
    return out_dir / f"layers_{layer_tag}_sanity.pt"


def default_plot_path(output_pt_path):
    return output_pt_path.with_suffix(".png")


def plot_summary(summary, args, plot_path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Please install it first, e.g. pip install matplotlib"
        ) from exc

    if not summary["budgets"]:
        print("[WARN] No evaluated budget data available. Skip plotting.")
        return

    budgets = sorted(float(k) for k in summary["budgets"].keys())
    metric_to_values = {m: [] for m in args.plot_metrics}
    metric_to_errors = {m: [] for m in args.plot_metrics}

    for b in budgets:
        entry = summary["budgets"][b]
        for m in args.plot_metrics:
            metric_to_values[m].append(float(entry[m]))
            if args.error_bar == "std":
                metric_to_errors[m].append(float(entry.get(f"{m}_std", 0.0)))
            elif args.error_bar == "sem":
                metric_to_errors[m].append(float(entry.get(f"{m}_sem", 0.0)))
            else:
                metric_to_errors[m].append(0.0)

    plt.figure(figsize=(10, 6))
    for m in args.plot_metrics:
        if args.error_bar == "none":
            plt.plot(budgets, metric_to_values[m], marker="o", linewidth=2, label=m)
        else:
            plt.errorbar(
                budgets,
                metric_to_values[m],
                yerr=metric_to_errors[m],
                marker="o",
                linewidth=2,
                capsize=3,
                label=f"{m} ({args.error_bar})",
            )

    plt.xscale("log")
    if args.plot_logy:
        plt.yscale("log")

    plt.xlabel("budget")
    plt.ylabel("metric value")
    plt.title(
        f"Multi-layer sanity metrics vs budget (layers={summary['layers']}, "
        f"dataset={summary['dataset']}, strategy={summary['strategy']})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=160)
    print(f"Saved plot to: {plot_path}")

    if not args.no_show:
        plt.show()


def main():
    args = parse_args()
    set_seed(42)

    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    validate_args_with_cache(ctx, args)

    ctx.model.eval()
    if args.all_layers:
        target_layers = list(range(ctx.model_config.num_hidden_layers))
    elif args.layers is None:
        target_layers = [5, 10, 15, 20, 25, 30]
    else:
        target_layers = args.layers

    n_heads = ctx.model_config.num_attention_heads
    head_idx = list(range(n_heads))
    pos_list = list(range(args.seq_len - args.tail_len, args.seq_len))

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)

    with torch.no_grad():
        ref_tail_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()

    script_dir = Path(__file__).resolve().parent.parent
    llama_dir = script_dir.parent
    result_root = llama_dir / "result"

    summary = {
        "layers": target_layers,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "loss_type": args.loss_type,
        "budgets": {},
    }

    for budget in args.budgets:
        layer_to_patch = {}
        try:
            for layer_idx in target_layers:
                alpha = get_alpha_for_layer_budget(
                    result_root=result_root,
                    layer_idx=layer_idx,
                    dataset=args.dataset,
                    strategy=args.strategy,
                    loss_type=args.loss_type,
                    budget=budget,
                )
                patch_hidden = build_modified_attn_hidden(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=alpha,
                    device=ctx.device,
                )
                layer_to_patch[layer_idx] = patch_hidden
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[WARN] Skip budget {budget}: {exc}")
            continue

        student_tail_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=layer_to_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        metrics = compute_metrics(ref_tail_logits, student_tail_logits, labels)
        summary["budgets"][float(budget)] = metrics

        print(
            f"[multi-layer] layers={target_layers}, budget={budget}, loss={args.loss_type}: "
            f"KL={metrics['sanity_kl']:.6f}, "
            f"teacher NLL={metrics['teacher_nll']:.6f}, "
            f"student NLL={metrics['student_nll']:.6f}, "
            f"NLL gap={metrics['nll_gap']:.6f}"
        )

    out_path = Path(args.output_path) if args.output_path else default_output_path(
        args.dataset, args.strategy, args.loss_type, target_layers
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(summary, out_path)
    print(f"Saved multi-layer sanity summary to: {out_path}")

    if args.plot:
        image_path = Path(args.plot_path) if args.plot_path else default_plot_path(out_path)
        plot_summary(summary, args, image_path)


if __name__ == "__main__":
    main()
