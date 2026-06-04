import argparse
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
            "Sample a few heads and plot |V_i| across token positions for one Llama "
            "layer. This runs a fresh forward from checkpoint and dataset instead of "
            "loading pre-saved KV tensors."
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
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_v_heads, seq_len):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be > 0")
    if args.pos_start < 0 or args.pos_start >= seq_len:
        raise ValueError(f"Invalid --pos-start {args.pos_start}; expected [0, {seq_len - 1}]")

    pos_end = seq_len if args.pos_end is None else min(args.pos_end, seq_len)
    if pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start after clipping to seq_len")

    if args.heads is not None:
        bad = [h for h in args.heads if h < 0 or h >= num_v_heads]
        if bad:
            raise ValueError(f"Invalid --heads entries {bad}; expected [0, {num_v_heads - 1}]")

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
        output_dir = f"../result/{args.dataset}_{args.start}/analyse_v/layer{args.layer}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def v_norm_for_head(v_all, head_idx, pos_start, pos_end):
    v = v_all[0, head_idx, pos_start:pos_end].float()
    return torch.norm(v, p=2, dim=-1).detach().cpu()


def summarize_norms(values):
    values = values.detach().float().cpu()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "n_tokens": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(values.max().item()),
    }


def save_stats_tsv(path, rows):
    columns = [
        "head",
        "pos_start",
        "pos_end",
        "n_tokens",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "max",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def save_values_tsv(path, pos_ids, heads, norms_by_head):
    with open(path, "w", encoding="utf-8") as f:
        f.write("pos\thead\tv_norm\n")
        for head in heads:
            values = norms_by_head[head]
            for pos, value in zip(pos_ids, values.tolist()):
                f.write(f"{pos}\t{head}\t{value:.8e}\n")


def plot_v_norms(out_path, layer_idx, pos_ids, heads, norms_by_head, dpi):
    n_heads = len(heads)
    fig_h = max(2.8 * n_heads, 3.2)
    fig, axes = plt.subplots(
        n_heads,
        1,
        figsize=(11.0, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for row, head in enumerate(heads):
        ax = axes[row][0]
        values = norms_by_head[head].detach().float().cpu().numpy()
        ax.plot(pos_ids, values, linewidth=1.0, color="#4C78A8")
        ax.scatter(pos_ids, values, s=5, alpha=0.55, color="#4C78A8")
        ax.axhline(values.mean(), color="#F58518", linewidth=1.0, label="mean")
        ax.set_title(f"L{layer_idx} H{head}", fontsize=10)
        ax.set_xlabel("id")
        ax.set_ylabel("|V_i|")
        ax.grid(alpha=0.22)
        if row == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"V norm by token id, layer {layer_idx}", fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = str_to_torch_dtype(args.dtype)

    from analysis.online_routing import capture_layer_artifacts
    from analysis.runner import load_context
    from analysis.sanity import move_model_inputs_to_device

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    input_seq_len = ctx.inputs["input_ids"].shape[1]
    if args.layer < 0 or args.layer >= ctx.model_config.num_hidden_layers:
        raise ValueError(
            f"Invalid --layer {args.layer}; expected [0, {ctx.model_config.num_hidden_layers - 1}]"
        )

    print(f"Selected layer: {args.layer}")
    print("Running one forward pass to capture V from checkpoint + dataset...")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    capture_pos = [min(max(args.pos_start, 0), input_seq_len - 1)]
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=capture_pos,
        model_inputs=model_inputs,
    )
    v_all = artifacts["v"]
    num_v_heads = v_all.shape[1]

    pos_end = validate_args(
        args=args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_v_heads=num_v_heads,
        seq_len=input_seq_len,
    )
    heads = sample_or_use_explicit(
        args.heads,
        list(range(num_v_heads)),
        args.num_heads,
        rng,
    )
    pos_ids = list(range(args.pos_start, pos_end))

    print(f"Selected V heads: {heads}")
    print(f"Selected id range: [{args.pos_start}, {pos_end})")

    norms_by_head = {}
    stat_rows = []
    for head in heads:
        norms = v_norm_for_head(
            v_all=v_all,
            head_idx=head,
            pos_start=args.pos_start,
            pos_end=pos_end,
        )
        norms_by_head[head] = norms
        stats = summarize_norms(norms)
        row = {
            "head": head,
            "pos_start": args.pos_start,
            "pos_end": pos_end,
            **stats,
        }
        stat_rows.append(row)
        print(
            "H{head:02d} ids=[{pos_start},{pos_end}) n={n_tokens} "
            "mean={mean:.4f} std={std:.4f} p05={p05:.4f} "
            "p50={p50:.4f} p95={p95:.4f} max={max:.4f}".format(**row)
        )

    output_dir = resolve_output_dir(args)
    plot_path = os.path.join(output_dir, "v_norm_by_id.png")
    stats_path = os.path.join(output_dir, "v_norm_stats.tsv")
    values_path = os.path.join(output_dir, "v_norm_values.tsv")

    plot_v_norms(
        out_path=plot_path,
        layer_idx=args.layer,
        pos_ids=pos_ids,
        heads=heads,
        norms_by_head=norms_by_head,
        dpi=args.plot_dpi,
    )
    save_stats_tsv(stats_path, stat_rows)
    save_values_tsv(values_path, pos_ids, heads, norms_by_head)

    print(f"Saved plot to {plot_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved values to {values_path}")


if __name__ == "__main__":
    main()
