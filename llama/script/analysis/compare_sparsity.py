import matplotlib.pyplot as plt
import torch


def summarize_sparsity_per_head(weights, pos_list, thresholds, mass_levels):
    """Summarize sparsity for [n_heads, n_pos, seq_len] weights on valid causal prefixes."""
    eps = 1e-12
    n_heads = weights.shape[0]
    out = {}

    thresholds = [float(x) for x in thresholds]
    mass_levels = sorted(float(x) for x in mass_levels)

    for h in range(n_heads):
        density_acc = {thr: [] for thr in thresholds}
        k_ratio_acc = {lvl: [] for lvl in mass_levels}
        entropy_acc = []

        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            if total_available <= 0:
                continue

            row = weights[h, row_i, :total_available].detach().float()

            for thr in thresholds:
                density_acc[thr].append((row > thr).float().mean().item())

            sorted_row = torch.sort(row, descending=True).values
            cumsum = torch.cumsum(sorted_row, dim=0)
            for lvl in mass_levels:
                idx = int(torch.searchsorted(cumsum, torch.tensor(lvl, device=row.device)).item())
                k = min(idx + 1, total_available)
                k_ratio_acc[lvl].append(k / float(total_available))

            entropy = -(row * torch.log(row.clamp_min(eps))).sum().item()
            entropy_norm = entropy / max(torch.log(torch.tensor(float(total_available))).item(), eps)
            entropy_acc.append(entropy_norm)

        head_stat = {
            "num_positions": len(entropy_acc),
            "entropy_norm_mean": float(torch.tensor(entropy_acc).mean().item()),
        }
        for thr in thresholds:
            key = f"density_gt_{thr:g}"
            head_stat[key] = float(torch.tensor(density_acc[thr]).mean().item())
        for lvl in mass_levels:
            key = f"k_ratio_for_mass_{lvl:g}"
            head_stat[key] = float(torch.tensor(k_ratio_acc[lvl]).mean().item())

        out[h] = head_stat

    return out


def summarize_sparsity_global(per_head_stats):
    keys = list(next(iter(per_head_stats.values())).keys())
    numeric_keys = [k for k in keys if k != "num_positions"]
    out = {"num_heads": len(per_head_stats)}
    for k in numeric_keys:
        vals = torch.tensor([float(v[k]) for v in per_head_stats.values()], dtype=torch.float32)
        out[f"{k}_mean"] = float(vals.mean().item())
        out[f"{k}_std"] = float(vals.std(unbiased=False).item())
    out["num_positions_mean"] = float(
        torch.tensor([float(v["num_positions"]) for v in per_head_stats.values()]).mean().item()
    )
    return out


def print_sparsity_report(head_labels, qk_raw_stats, qk_routing_stats, optimal_stats, mass_levels):
    first_lvl = float(sorted(mass_levels)[0])
    first_density_key = "density_gt_0.001"
    first_k_key = f"k_ratio_for_mass_{first_lvl:g}"

    print("===== Sparsity Check (Per Head) =====")
    print(
        "Columns: head | qk_raw dens>1e-3 | qk_route dens>1e-3 | opt dens>1e-3 | "
        f"qk_raw k@{first_lvl:g} | qk_route k@{first_lvl:g} | opt k@{first_lvl:g}"
    )

    for i, h in enumerate(head_labels):
        s_raw = qk_raw_stats[i]
        s_route = qk_routing_stats[i]
        s_opt = optimal_stats[i]
        print(
            f"head {h:>2d} | "
            f"{s_raw[first_density_key]:.4f} | "
            f"{s_route[first_density_key]:.4f} | "
            f"{s_opt[first_density_key]:.4f} | "
            f"{s_raw[first_k_key]:.4f} | "
            f"{s_route[first_k_key]:.4f} | "
            f"{s_opt[first_k_key]:.4f} |"
            f"{s_opt[first_k_key]-s_route[first_k_key]:.4f}"
        )


def compute_per_pos_sparsity_curves(weights, pos_list, mass_level=0.9):
    """Return per-position entropy_norm and k-ratio curves for [n_heads, n_pos, seq_len]."""
    eps = 1e-12
    n_heads = weights.shape[0]
    curves = {}

    for h in range(n_heads):
        entropy_curve = []
        k_ratio_curve = []

        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            row = weights[h, row_i, :total_available].detach().float()

            entropy = -(row * torch.log(row.clamp_min(eps))).sum().item()
            entropy_norm = entropy / max(torch.log(torch.tensor(float(total_available))).item(), eps)

            sorted_row = torch.sort(row, descending=True).values
            cumsum = torch.cumsum(sorted_row, dim=0)
            idx = int(torch.searchsorted(cumsum, torch.tensor(float(mass_level), device=row.device)).item())
            k = min(idx + 1, total_available)

            entropy_curve.append(float(entropy_norm))
            k_ratio_curve.append(float(k))

        curves[h] = {
            "entropy_norm": entropy_curve,
            "k_ratio": k_ratio_curve,
        }

    return curves


def plot_sparsity_curves_grid(
    head_labels,
    pos_list,
    qk_raw_curves,
    qk_routing_curves,
    optimal_curves,
    out_path,
    dpi,
    mass_level,
):
    n_heads = len(head_labels)
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 2)

    xs = list(pos_list)
    for i, head_label in enumerate(head_labels):
        c_raw = qk_raw_curves[i]
        c_route = qk_routing_curves[i]
        c_opt = optimal_curves[i]

        ax0 = axes[i, 0]
        # ax0.plot(xs, c_raw["entropy_norm"], label="qk_raw", linewidth=1.1)
        ax0.plot(xs, c_route["entropy_norm"], label="qk_routing", linewidth=1.2)
        ax0.plot(xs, c_opt["entropy_norm"], label="optimal", linewidth=1.2)
        ax0.set_ylim(0.0, 1.0)
        ax0.set_xlabel("Position")
        ax0.set_ylabel("Entropy (normalized)")
        ax0.set_title(f"Head {head_label}: per-pos entropy")
        ax0.grid(True, linestyle="--", alpha=0.35)
        ax0.legend()

        ax1 = axes[i, 1]
        # ax1.plot(xs, c_raw["k_ratio"], label="qk_raw", linewidth=1.1)
        ax1.plot(xs, c_route["k_ratio"], label="qk_routing", linewidth=1.2)
        ax1.plot(xs, c_opt["k_ratio"], label="optimal", linewidth=1.2)
        # ax1.set_ylim(0.0, 0.2)
        # ax1.set_yscale("log")
        ax1.set_xlabel("Position")
        ax1.set_ylabel(f"k-ratio for mass {mass_level:g}")
        ax1.set_title(f"Head {head_label}: per-pos k@{mass_level:g}")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
