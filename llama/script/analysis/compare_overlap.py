import matplotlib.pyplot as plt
import torch


def compute_overlap_stats(alpha_opt, alpha_base, pos_list, topk):
    # alpha_*: [n_heads, n_pos, seq_len]
    per_head = {}
    per_head_summaries = {}

    for head_idx in range(alpha_opt.shape[0]):
        per_pos = []
        for row_i, pos in enumerate(pos_list):
            total_available = pos + 1
            k = min(topk, total_available)
            if k <= 0:
                continue

            opt_row = alpha_opt[head_idx, row_i, :total_available]
            base_row = alpha_base[head_idx, row_i, :total_available]

            opt_idx = torch.topk(opt_row, k=k, largest=True).indices
            base_idx = torch.topk(base_row, k=k, largest=True).indices

            opt_set = set(opt_idx.detach().cpu().tolist())
            base_set = set(base_idx.detach().cpu().tolist())
            inter = opt_set.intersection(base_set)
            union = opt_set.union(base_set)

            overlap_ratio = len(inter) / float(k)
            jaccard = len(inter) / float(len(union)) if len(union) > 0 else 1.0

            cross_mass_opt_on_base = opt_row[base_idx].sum().item()
            cross_mass_base_on_opt = base_row[opt_idx].sum().item()

            per_pos.append(
                {
                    "pos": int(pos),
                    "k": int(k),
                    "overlap_ratio": float(overlap_ratio),
                    "jaccard": float(jaccard),
                    "opt_mass_on_base_topk": float(cross_mass_opt_on_base),
                    "base_mass_on_opt_topk": float(cross_mass_base_on_opt),
                }
            )

        if len(per_pos) == 0:
            raise ValueError("No valid position for overlap statistics.")

        overlap_values = torch.tensor([x["overlap_ratio"] for x in per_pos], dtype=torch.float32)
        jaccard_values = torch.tensor([x["jaccard"] for x in per_pos], dtype=torch.float32)
        opt_on_base_values = torch.tensor(
            [x["opt_mass_on_base_topk"] for x in per_pos], dtype=torch.float32
        )
        base_on_opt_values = torch.tensor(
            [x["base_mass_on_opt_topk"] for x in per_pos], dtype=torch.float32
        )

        summary = {
            "mean_overlap_ratio": overlap_values.mean().item(),
            "std_overlap_ratio": overlap_values.std(unbiased=False).item(),
            "mean_jaccard": jaccard_values.mean().item(),
            "std_jaccard": jaccard_values.std(unbiased=False).item(),
            "mean_opt_mass_on_base_topk": opt_on_base_values.mean().item(),
            "mean_base_mass_on_opt_topk": base_on_opt_values.mean().item(),
            "num_positions": len(per_pos),
        }
        per_head[head_idx] = per_pos
        per_head_summaries[head_idx] = summary

    mean_overlap = torch.tensor(
        [v["mean_overlap_ratio"] for v in per_head_summaries.values()], dtype=torch.float32
    )
    mean_jaccard = torch.tensor(
        [v["mean_jaccard"] for v in per_head_summaries.values()], dtype=torch.float32
    )
    global_summary = {
        "mean_overlap_ratio": mean_overlap.mean().item(),
        "std_overlap_ratio": mean_overlap.std(unbiased=False).item(),
        "mean_jaccard": mean_jaccard.mean().item(),
        "std_jaccard": mean_jaccard.std(unbiased=False).item(),
        "num_heads": len(per_head_summaries),
    }
    return global_summary, per_head_summaries, per_head


def summarize_diff_v_topk_per_head(alpha_opt, alpha_base, v_abs, pos_list, topk=10):
    """Per-head signed top-k for diff_v, with decomposed terms.

    diff_v = (alpha_opt - alpha_base) * |V|
    alpha_*: [n_heads, n_pos, seq_len]
    v_abs:   [n_heads, seq_len]
    """
    if alpha_opt.shape != alpha_base.shape:
        raise ValueError(
            f"Shape mismatch: alpha_opt={tuple(alpha_opt.shape)} vs alpha_base={tuple(alpha_base.shape)}"
        )
    if v_abs.ndim != 2:
        raise ValueError(f"v_abs must have shape [n_heads, seq_len], got {tuple(v_abs.shape)}")
    if alpha_opt.shape[0] != v_abs.shape[0] or alpha_opt.shape[2] != v_abs.shape[1]:
        raise ValueError(
            "Shape mismatch for diff_v top-k: "
            f"alpha shape={tuple(alpha_opt.shape)}, v_abs shape={tuple(v_abs.shape)}"
        )
    if topk <= 0:
        raise ValueError("topk must be > 0")

    delta_alpha = (alpha_opt - alpha_base).detach().float()
    v_abs = v_abs.detach().float()
    diff_v = delta_alpha * v_abs.unsqueeze(1)

    n_heads = diff_v.shape[0]
    n_pos = diff_v.shape[1]
    seq_len = diff_v.shape[2]
    out = {}

    for h in range(n_heads):
        valid_mask = torch.zeros((n_pos, seq_len), dtype=torch.bool, device=diff_v.device)
        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            if total_available <= 0:
                continue
            valid_mask[row_i, : min(total_available, seq_len)] = True

        if not valid_mask.any():
            raise ValueError("No valid entries for abs diff statistics.")

        signed_vals = diff_v[h][valid_mask]
        abs_vals = signed_vals.abs()
        coords = valid_mask.nonzero(as_tuple=False)

        k = min(int(topk), int(abs_vals.numel()))

        pos_vals, pos_idx = torch.topk(signed_vals, k=k, largest=True)
        neg_vals, neg_idx = torch.topk(signed_vals, k=k, largest=False)

        top_pos = []
        for rank in range(k):
            coord = coords[pos_idx[rank]]
            row_i = int(coord[0].item())
            key_pos = int(coord[1].item())
            score_v = float(pos_vals[rank].item())
            attn_v = float(delta_alpha[h, row_i, key_pos].item())
            v_norm = float(v_abs[h, key_pos].item())
            top_pos.append(
                {
                    "rank": rank + 1,
                    "query_pos": int(pos_list[row_i]),
                    "key_pos": key_pos,
                    "diff_v_score": score_v,
                    "attention_score": attn_v,
                    "v_abs": v_norm,
                }
            )

        top_neg = []
        for rank in range(k):
            coord = coords[neg_idx[rank]]
            row_i = int(coord[0].item())
            key_pos = int(coord[1].item())
            score_v = float(neg_vals[rank].item())
            attn_v = float(delta_alpha[h, row_i, key_pos].item())
            v_norm = float(v_abs[h, key_pos].item())
            top_neg.append(
                {
                    "rank": rank + 1,
                    "query_pos": int(pos_list[row_i]),
                    "key_pos": key_pos,
                    "diff_v_score": score_v,
                    "attention_score": attn_v,
                    "v_abs": v_norm,
                }
            )

        out[h] = {
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    return out


def print_signed_topk_report(head_labels, signed_topk, score_name):
    print(f"===== {score_name} top positive/negative (Per Head) =====")
    print("Columns: head | rank | query_pos | key_pos | diff_v_score | attention_score | |V|")
    for i, h in enumerate(head_labels):
        print(f"-- head {h:>2d} positive top-k --")
        for item in signed_topk[i]["top_positive"]:
            print(
                f"head {h:>2d} | "
                f"{item['rank']:>1d} | "
                f"{item['query_pos']:>4d} | "
                f"{item['key_pos']:>4d} | "
                f"{item['diff_v_score']:+.6e} | "
                f"{item['attention_score']:+.6e} | "
                f"{item['v_abs']:.6e}"
            )
        print(f"-- head {h:>2d} negative top-k --")
        for item in signed_topk[i]["top_negative"]:
            print(
                f"head {h:>2d} | "
                f"{item['rank']:>1d} | "
                f"{item['query_pos']:>4d} | "
                f"{item['key_pos']:>4d} | "
                f"{item['diff_v_score']:+.6e} | "
                f"{item['attention_score']:+.6e} | "
                f"{item['v_abs']:.6e}"
            )


def build_diff_v_map(alpha_opt: torch.Tensor, alpha_base: torch.Tensor, v_abs: torch.Tensor):
    """Build delta-alpha weighted by |V| for visualization.

    alpha_*: [n_heads, n_pos, seq_len]
    v_abs:   [n_heads, seq_len]
    """
    if alpha_opt.shape != alpha_base.shape:
        raise ValueError(
            f"Shape mismatch: alpha_opt={tuple(alpha_opt.shape)} vs alpha_base={tuple(alpha_base.shape)}"
        )
    if v_abs.ndim != 2:
        raise ValueError(f"v_abs must have shape [n_heads, seq_len], got {tuple(v_abs.shape)}")
    if alpha_opt.shape[0] != v_abs.shape[0] or alpha_opt.shape[2] != v_abs.shape[1]:
        raise ValueError(
            "Shape mismatch for diff_v: "
            f"alpha shape={tuple(alpha_opt.shape)}, v_abs shape={tuple(v_abs.shape)}"
        )

    delta_alpha = alpha_opt - alpha_base
    return delta_alpha * v_abs.unsqueeze(1)


def plot_overlap_grid(per_head, head_labels, out_path, dpi):
    n_heads = len(head_labels)
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 2)

    for i, head_label in enumerate(head_labels):
        per_pos = per_head[i]
        positions = [x["pos"] for x in per_pos]
        overlap = [x["overlap_ratio"] for x in per_pos]
        jaccard = [x["jaccard"] for x in per_pos]
        mass1 = [x["opt_mass_on_base_topk"] for x in per_pos]
        mass2 = [x["base_mass_on_opt_topk"] for x in per_pos]

        axes[i, 0].plot(positions, overlap, label="top-k overlap ratio", linewidth=1.4)
        axes[i, 0].plot(positions, jaccard, label="jaccard", linewidth=1.2)
        axes[i, 0].set_ylim(0.0, 1.0)
        axes[i, 0].set_xlabel("Position")
        axes[i, 0].set_ylabel("Set overlap")
        axes[i, 0].set_title(f"Head {head_label}: overlap")
        axes[i, 0].grid(True, linestyle="--", alpha=0.35)
        axes[i, 0].legend()

        axes[i, 1].plot(positions, mass1, label="opt mass on base top-k", linewidth=1.4)
        axes[i, 1].plot(positions, mass2, label="base mass on opt top-k", linewidth=1.4)
        axes[i, 1].set_ylim(0.0, 1.0)
        axes[i, 1].set_xlabel("Position")
        axes[i, 1].set_ylabel("Cross mass")
        axes[i, 1].set_title(f"Head {head_label}: cross-mass")
        axes[i, 1].grid(True, linestyle="--", alpha=0.35)
        axes[i, 1].legend()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
