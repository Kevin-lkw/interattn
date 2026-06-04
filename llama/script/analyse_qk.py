import argparse
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analysis.config import set_seed, str_to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample a few heads and query positions, then plot the distribution of "
            "QK / sqrt(head_dim) logits for one Llama layer. This runs a fresh forward "
            "from checkpoint and dataset instead of loading pre-saved KV tensors."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--queries", type=int, nargs="+", default=None)
    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads, seq_len):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be > 0")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.bins <= 0:
        raise ValueError("--bins must be > 0")
    if args.pos_start < 0 or args.pos_start >= seq_len:
        raise ValueError(f"Invalid --pos-start {args.pos_start}; expected [0, {seq_len - 1}]")

    pos_end = seq_len if args.pos_end is None else min(args.pos_end, seq_len)
    if pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start after clipping to seq_len")

    if args.heads is not None:
        bad = [h for h in args.heads if h < 0 or h >= num_heads]
        if bad:
            raise ValueError(f"Invalid --heads entries {bad}; expected [0, {num_heads - 1}]")
    if args.queries is not None:
        bad = [q for q in args.queries if q < 0 or q >= seq_len]
        if bad:
            raise ValueError(f"Invalid --queries entries {bad}; expected [0, {seq_len - 1}]")

    return pos_end


def sample_or_use_explicit(explicit, candidates, n, rng):
    if explicit is not None and len(explicit) > 0:
        return sorted(set(int(x) for x in explicit))
    n = min(n, len(candidates))
    return sorted(rng.sample(candidates, n))


def resolve_output_dir(args):
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"../result/{args.dataset}_{args.start}/analyse_qk/layer{args.layer}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def qk_logits_for_query(layer_ctx, layer_idx, head_idx, query_pos, device):
    q_all = layer_ctx.rope_qkv[layer_idx]["q"][0]
    k_all = layer_ctx.rope_qkv[layer_idx]["k"][0]

    q = q_all[head_idx, query_pos].to(device=device, dtype=torch.float32)
    k = k_all[head_idx, : query_pos + 1].to(device=device, dtype=torch.float32)
    return (k @ q) / math.sqrt(q.numel())


def summarize_logits(values):
    values = values.detach().float().cpu()
    exp_values = values.exp()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32),
    )
    exp_quantiles = torch.quantile(
        exp_values,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "n_keys": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(values.max().item()),
        "exp_mean": float(exp_values.mean().item()),
        "exp_std": float(exp_values.std(unbiased=False).item()),
        "exp_min": float(exp_values.min().item()),
        "exp_p01": float(exp_quantiles[0].item()),
        "exp_p05": float(exp_quantiles[1].item()),
        "exp_p50": float(exp_quantiles[2].item()),
        "exp_p95": float(exp_quantiles[3].item()),
        "exp_p99": float(exp_quantiles[4].item()),
        "exp_max": float(exp_values.max().item()),
    }


def save_stats_tsv(path, rows):
    columns = [
        "head",
        "query_pos",
        "n_keys",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "max",
        "exp_mean",
        "exp_std",
        "exp_min",
        "exp_p01",
        "exp_p05",
        "exp_p50",
        "exp_p95",
        "exp_p99",
        "exp_max",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def plot_distributions(
    out_path,
    layer_idx,
    heads,
    queries,
    logits_by_pair,
    bins,
    dpi,
    transform,
    xlabel,
    title,
    color,
    xscale=None,
):
    n_rows = len(heads)
    n_cols = len(queries)
    fig_w = max(4.0 * n_cols, 5.0)
    fig_h = max(2.8 * n_rows, 3.2)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for r, head in enumerate(heads):
        for c, query_pos in enumerate(queries):
            ax = axes[r][c]
            raw = logits_by_pair[(head, query_pos)].detach().float().cpu()
            values = transform(raw)
            values_np = values.numpy()
            ax.hist(values_np, bins=bins, density=True, alpha=0.82, color=color)
            ax.axvline(values.mean(), color="#F58518", linewidth=1.2, label="mean")
            ax.axvline(float(values.median()), color="#54A24B", linewidth=1.2, label="median")
            ax.set_title(f"L{layer_idx} H{head} Q{query_pos}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            if xscale is not None:
                ax.set_xscale(xscale)
            ax.grid(alpha=0.22)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"{title}, layer {layer_idx}", fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = str_to_torch_dtype(args.dtype)

    from analysis.online_routing import build_runtime_layer_ctx, capture_layer_artifacts
    from analysis.runner import load_context
    from analysis.sanity import move_model_inputs_to_device

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    input_seq_len = ctx.inputs["input_ids"].shape[1]
    pos_end = validate_args(
        args=args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
        seq_len=input_seq_len,
    )

    heads = sample_or_use_explicit(
        args.heads,
        list(range(ctx.model_config.num_attention_heads)),
        args.num_heads,
        rng,
    )
    queries = sample_or_use_explicit(
        args.queries,
        list(range(args.pos_start, pos_end)),
        args.num_queries,
        rng,
    )
    pos_list = queries

    print(f"Selected layer: {args.layer}")
    print(f"Selected heads: {heads}")
    print(f"Selected query positions: {queries}")
    print("Running one forward pass to capture Q/K/V from checkpoint + dataset...")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    logits_by_pair = {}
    stat_rows = []
    for head in heads:
        for query_pos in queries:
            logits = qk_logits_for_query(
                layer_ctx=layer_ctx,
                layer_idx=args.layer,
                head_idx=head,
                query_pos=query_pos,
                device=ctx.device,
            )
            logits_by_pair[(head, query_pos)] = logits.cpu()
            stats = summarize_logits(logits)
            row = {"head": head, "query_pos": query_pos, **stats}
            stat_rows.append(row)
            print(
                "H{head:02d} Q{query_pos:04d} n={n_keys} "
                "mean={mean:.4f} std={std:.4f} p05={p05:.4f} "
                "p50={p50:.4f} p95={p95:.4f} "
                "exp_mean={exp_mean:.4e} exp_p50={exp_p50:.4e} "
                "exp_p95={exp_p95:.4e}".format(**row)
            )

    output_dir = resolve_output_dir(args)
    plot_path = os.path.join(output_dir, "qk_distribution_grid.png")
    exp_plot_path = os.path.join(output_dir, "exp_qk_distribution_grid.png")
    stats_path = os.path.join(output_dir, "qk_distribution_stats.tsv")

    plot_distributions(
        out_path=plot_path,
        layer_idx=args.layer,
        heads=heads,
        queries=queries,
        logits_by_pair=logits_by_pair,
        bins=args.bins,
        dpi=args.plot_dpi,
        transform=lambda x: x,
        xlabel="qk / sqrt(d)",
        title="QK / sqrt(d) distribution",
        color="#4C78A8",
    )
    plot_distributions(
        out_path=exp_plot_path,
        layer_idx=args.layer,
        heads=heads,
        queries=queries,
        logits_by_pair=logits_by_pair,
        bins=args.bins,
        dpi=args.plot_dpi,
        transform=torch.exp,
        xlabel="exp(qk / sqrt(d))",
        title="exp(QK / sqrt(d)) distribution",
        color="#72B7B2",
        xscale="log",
    )
    save_stats_tsv(stats_path, stat_rows)

    print(f"Saved plot to {plot_path}")
    print(f"Saved exp plot to {exp_plot_path}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
