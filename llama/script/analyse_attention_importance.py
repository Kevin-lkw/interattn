import argparse
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from analysis.config import set_seed, str_to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention-score importance for one Llama layer. "
            "For each query, important keys are the smallest top-alpha set covering "
            "a target attention mass. The script also counts how many queries mark "
            "each key as important."
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
    parser.add_argument(
        "--mass-levels",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Attention-mass levels used to define important top-p key sets.",
    )
    parser.add_argument(
        "--primary-mass-level",
        type=float,
        default=0.95,
        help="Mass level highlighted in key-hit plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads, seq_len):
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
        bad = [h for h in args.heads if h < 0 or h >= num_heads]
        if bad:
            raise ValueError(f"Invalid --heads entries {bad}; expected [0, {num_heads - 1}]")

    if len(args.mass_levels) == 0:
        raise ValueError("--mass-levels must contain at least one level")
    for level in args.mass_levels:
        if level <= 0.0 or level > 1.0:
            raise ValueError(f"Invalid mass level {level}; expected (0, 1]")

    mass_levels = sorted(set(float(x) for x in args.mass_levels))
    primary = float(args.primary_mass_level)
    if primary not in mass_levels:
        mass_levels.append(primary)
        mass_levels = sorted(mass_levels)
    return pos_end, mass_levels, primary


def sample_or_use_explicit(explicit, candidates, n, rng):
    if explicit is not None and len(explicit) > 0:
        return sorted(set(int(x) for x in explicit))
    n = min(n, len(candidates))
    return sorted(rng.sample(candidates, n))


def resolve_output_dir(args):
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"../result/{args.dataset}_{args.start}/analyse_attention_importance/layer{args.layer}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def mass_tag(level):
    return f"m{int(round(level * 1000)):03d}"


def compute_attention_alpha(q_all, k_all, heads, pos_list, device):
    q = q_all[0, heads][:, pos_list, :].to(device=device, dtype=torch.float32)
    k = k_all[0, heads].to(device=device, dtype=torch.float32)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    seq_len = scores.shape[-1]
    causal = torch.full((len(pos_list), seq_len), float("-inf"), device=device)
    for i, pos in enumerate(pos_list):
        causal[i, : pos + 1] = 0.0
    return F.softmax(scores + causal.unsqueeze(0), dim=-1)


def summarize_alpha(alpha, heads, pos_list, seq_len, mass_levels):
    query_rows = []
    key_rows = []
    summary_rows = []

    n_heads = len(heads)
    n_levels = len(mass_levels)
    required_k = torch.zeros(n_heads, len(pos_list), n_levels, dtype=torch.long)
    hit_count = torch.zeros(n_heads, n_levels, seq_len, dtype=torch.long)
    visible_count = torch.zeros(len(pos_list), seq_len, dtype=torch.long)
    max_alpha = torch.zeros(n_heads, seq_len, dtype=torch.float32)
    sum_alpha = torch.zeros(n_heads, seq_len, dtype=torch.float32)

    for i, pos in enumerate(pos_list):
        visible_count[i, : pos + 1] = 1
    visible_per_key = visible_count.sum(dim=0)

    for h_ord, head in enumerate(heads):
        for i, pos in enumerate(pos_list):
            row = alpha[h_ord, i, : pos + 1].detach().float().cpu()
            sorted_vals, sorted_idx = torch.sort(row, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=0)
            entropy = -torch.sum(row * torch.log(row.clamp_min(1e-30)))

            q_row = {
                "head": int(head),
                "query_pos": int(pos),
                "n_keys": int(pos + 1),
                "entropy": float(entropy.item()),
                "max_alpha": float(sorted_vals[0].item()),
            }

            max_alpha[h_ord, : pos + 1] = torch.maximum(max_alpha[h_ord, : pos + 1], row)
            sum_alpha[h_ord, : pos + 1] += row

            for level_idx, level in enumerate(mass_levels):
                k_req = int(torch.searchsorted(cumsum, torch.tensor(level), right=False).item()) + 1
                k_req = min(k_req, pos + 1)
                required_k[h_ord, i, level_idx] = k_req
                important_idx = sorted_idx[:k_req]
                hit_count[h_ord, level_idx, important_idx] += 1
                tag = mass_tag(level)
                q_row[f"k_at_{tag}"] = k_req
                q_row[f"frac_at_{tag}"] = float(k_req / (pos + 1))

            query_rows.append(q_row)

    for h_ord, head in enumerate(heads):
        visible = visible_per_key > 0
        for key_pos in range(seq_len):
            n_visible = int(visible_per_key[key_pos].item())
            if n_visible == 0:
                continue
            row = {
                "head": int(head),
                "key_pos": int(key_pos),
                "visible_queries": n_visible,
                "max_alpha": float(max_alpha[h_ord, key_pos].item()),
                "mean_alpha_when_visible": float((sum_alpha[h_ord, key_pos] / n_visible).item()),
            }
            for level_idx, level in enumerate(mass_levels):
                hits = int(hit_count[h_ord, level_idx, key_pos].item())
                tag = mass_tag(level)
                row[f"hit_queries_{tag}"] = hits
                row[f"hit_ratio_{tag}"] = float(hits / n_visible)
            key_rows.append(row)

    for h_ord, head in enumerate(heads):
        for level_idx, level in enumerate(mass_levels):
            tag = mass_tag(level)
            k_values = required_k[h_ord, :, level_idx].float()
            frac_values = torch.tensor(
                [
                    required_k[h_ord, i, level_idx].item() / (pos + 1)
                    for i, pos in enumerate(pos_list)
                ],
                dtype=torch.float32,
            )
            hits = hit_count[h_ord, level_idx]
            visible = visible_per_key > 0
            covered = (hits[visible] > 0).float()
            summary_rows.append(
                {
                    "head": int(head),
                    "mass_level": float(level),
                    "tag": tag,
                    "queries": len(pos_list),
                    "visible_keys": int(visible.sum().item()),
                    "mean_k": float(k_values.mean().item()),
                    "median_k": float(torch.quantile(k_values, 0.5).item()),
                    "p95_k": float(torch.quantile(k_values, 0.95).item()),
                    "mean_frac": float(frac_values.mean().item()),
                    "median_frac": float(torch.quantile(frac_values, 0.5).item()),
                    "p95_frac": float(torch.quantile(frac_values, 0.95).item()),
                    "kv_covered": float(covered.mean().item()),
                    "mean_hit_queries_per_visible_key": float(hits[visible].float().mean().item()),
                }
            )

    for level_idx, level in enumerate(mass_levels):
        tag = mass_tag(level)
        k_values = required_k[:, :, level_idx].reshape(-1).float()
        frac_values = []
        for _h_ord in range(n_heads):
            for i, pos in enumerate(pos_list):
                frac_values.append(required_k[_h_ord, i, level_idx].item() / (pos + 1))
        frac_values = torch.tensor(frac_values, dtype=torch.float32)
        hits = hit_count[:, level_idx, :].sum(dim=0)
        visible = visible_per_key > 0
        covered = (hits[visible] > 0).float()
        summary_rows.append(
            {
                "head": "all",
                "mass_level": float(level),
                "tag": tag,
                "queries": len(pos_list) * n_heads,
                "visible_keys": int(visible.sum().item()),
                "mean_k": float(k_values.mean().item()),
                "median_k": float(torch.quantile(k_values, 0.5).item()),
                "p95_k": float(torch.quantile(k_values, 0.95).item()),
                "mean_frac": float(frac_values.mean().item()),
                "median_frac": float(torch.quantile(frac_values, 0.5).item()),
                "p95_frac": float(torch.quantile(frac_values, 0.95).item()),
                "kv_covered": float(covered.mean().item()),
                "mean_hit_queries_per_visible_key": float(hits[visible].float().mean().item()),
            }
        )

    return query_rows, key_rows, summary_rows, required_k, hit_count, visible_per_key


def save_tsv(path, rows, columns):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def plot_required_k(out_path, heads, pos_list, mass_levels, required_k, dpi):
    fig_h = max(2.8 * len(heads), 3.2)
    fig, axes = plt.subplots(
        len(heads),
        1,
        figsize=(11.0, fig_h),
        squeeze=False,
        constrained_layout=True,
    )
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    for h_ord, head in enumerate(heads):
        ax = axes[h_ord][0]
        for level_idx, level in enumerate(mass_levels):
            values = required_k[h_ord, :, level_idx].float().numpy()
            ax.plot(
                pos_list,
                values,
                linewidth=1.0,
                color=colors[level_idx % len(colors)],
                label=f"{level:g} mass",
            )
        ax.set_title(f"Head {head}: minimal full-attention KV count")
        ax.set_xlabel("query position")
        ax.set_ylabel("required keys")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_key_hits(out_path, heads, primary_level, hit_count, visible_per_key, dpi):
    primary_idx = None
    fig_h = max(2.8 * len(heads), 3.2)
    fig, axes = plt.subplots(
        len(heads),
        1,
        figsize=(11.0, fig_h),
        squeeze=False,
        constrained_layout=True,
    )
    x = torch.arange(visible_per_key.numel()).numpy()
    visible_np = visible_per_key.numpy()

    for h_ord, head in enumerate(heads):
        if primary_idx is None:
            primary_idx = 0
        hits = hit_count[h_ord, primary_idx].float().numpy()
        ax = axes[h_ord][0]
        ax.plot(x, hits, linewidth=1.0, color="#4C78A8", label="important-query count")
        ax.plot(x, visible_np, linewidth=0.8, color="#BAB0AC", alpha=0.8, label="visible-query count")
        ax.set_title(f"Head {head}: key hits in top-{primary_level:g} mass sets")
        ax.set_xlabel("key position")
        ax.set_ylabel("queries")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

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
    pos_end, mass_levels, primary = validate_args(
        args=args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
        seq_len=input_seq_len,
    )
    primary_idx = mass_levels.index(primary)

    heads = sample_or_use_explicit(
        args.heads,
        list(range(ctx.model_config.num_attention_heads)),
        args.num_heads,
        rng,
    )
    pos_list = list(range(args.pos_start, pos_end))

    print(f"Selected layer: {args.layer}")
    print(f"Selected heads: {heads}")
    print(f"Query positions: [{pos_list[0]}, {pos_list[-1]}], n={len(pos_list)}")
    print(f"Mass levels: {mass_levels}")
    print("Running one forward pass to capture Q/K from checkpoint + dataset...")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )

    alpha = compute_attention_alpha(
        q_all=artifacts["q"],
        k_all=artifacts["k"],
        heads=heads,
        pos_list=pos_list,
        device=ctx.device,
    )

    print("Summarizing per-query sparsity and per-key importance hits...")
    (
        query_rows,
        key_rows,
        summary_rows,
        required_k,
        hit_count,
        visible_per_key,
    ) = summarize_alpha(
        alpha=alpha,
        heads=heads,
        pos_list=pos_list,
        seq_len=input_seq_len,
        mass_levels=mass_levels,
    )

    output_dir = resolve_output_dir(args)
    query_path = os.path.join(output_dir, "query_required_k.tsv")
    key_path = os.path.join(output_dir, "key_importance_hits.tsv")
    summary_path = os.path.join(output_dir, "summary.tsv")
    k_plot_path = os.path.join(output_dir, "query_required_k.png")
    hit_plot_path = os.path.join(output_dir, f"key_hits_{mass_tag(primary)}.png")
    tensor_path = os.path.join(output_dir, "attention_importance_tensors.pt")

    query_columns = ["head", "query_pos", "n_keys", "entropy", "max_alpha"]
    for level in mass_levels:
        tag = mass_tag(level)
        query_columns += [f"k_at_{tag}", f"frac_at_{tag}"]

    key_columns = ["head", "key_pos", "visible_queries", "max_alpha", "mean_alpha_when_visible"]
    for level in mass_levels:
        tag = mass_tag(level)
        key_columns += [f"hit_queries_{tag}", f"hit_ratio_{tag}"]

    summary_columns = [
        "head",
        "mass_level",
        "tag",
        "queries",
        "visible_keys",
        "mean_k",
        "median_k",
        "p95_k",
        "mean_frac",
        "median_frac",
        "p95_frac",
        "kv_covered",
        "mean_hit_queries_per_visible_key",
    ]

    save_tsv(query_path, query_rows, query_columns)
    save_tsv(key_path, key_rows, key_columns)
    save_tsv(summary_path, summary_rows, summary_columns)
    plot_required_k(k_plot_path, heads, pos_list, mass_levels, required_k, args.plot_dpi)
    plot_key_hits(
        hit_plot_path,
        heads,
        primary,
        hit_count[:, primary_idx : primary_idx + 1, :],
        visible_per_key,
        args.plot_dpi,
    )
    torch.save(
        {
            "heads": heads,
            "pos_list": pos_list,
            "mass_levels": mass_levels,
            "primary_mass_level": primary,
            "required_k": required_k,
            "hit_count": hit_count,
            "visible_per_key": visible_per_key,
        },
        tensor_path,
    )

    print("Summary:")
    for row in summary_rows:
        if row["head"] == "all" or len(heads) <= 4:
            print(
                "head={head} mass={mass_level:g} mean_k={mean_k:.2f} "
                "p95_k={p95_k:.2f} mean_frac={mean_frac:.4f} "
                "kv_covered={kv_covered:.4f}".format(**row)
            )
    print(f"Saved per-query table to {query_path}")
    print(f"Saved per-key table to {key_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {k_plot_path} and {hit_plot_path}")
    print(f"Saved tensors to {tensor_path}")


if __name__ == "__main__":
    main()
