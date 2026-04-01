import matplotlib.pyplot as plt
import torch


def _to_plot_array(x: torch.Tensor):
    x = x.detach().float().cpu().clone()
    finite = torch.isfinite(x)
    if finite.any():
        min_val = x[finite].min().item()
        x[~finite] = min_val - 1.0
    else:
        x[:] = 0.0
    return x.numpy()


def build_prob_viz_map(probs_tensor: torch.Tensor, mode: str, eps: float = 1e-8):
    probs = probs_tensor.detach().float()
    probs = probs[:256, :256]
    if mode == "linear":
        return probs, "linear", "Reds", 0.0

    if mode == "log":
        # Log scale reveals tail structure in highly peaked rows.
        z = torch.log10(probs.clamp_min(eps))
        return z, f"log10 (eps={eps:g})", "viridis", None

    if mode == "row_log":
        # Normalize each row by its max before log; highlights relative pattern per query position.
        row_max = probs.max(dim=-1, keepdim=True).values.clamp_min(eps)
        z = torch.log10((probs / row_max).clamp_min(eps))
        return z, f"row-normalized log10 (eps={eps:g})", "viridis", None

    raise ValueError(f"Unknown viz mode: {mode}")


def signed_log_map(x: torch.Tensor, eps: float):
    x = x[:256, :256]
    return torch.sign(x) * torch.log10(1.0 + x.abs() / eps)


def plot_routing_grid(
    alpha_base,
    alpha_opt,
    diff_v,
    head_labels,
    out_path,
    dpi,
    alpha_viz,
    diff_log_eps,
):
    # alpha_*: [n_heads, n_pos, seq_len]
    n_heads = alpha_base.shape[0]
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 4, figsize=(20.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 4)

    for h in range(n_heads):
        base_map, alpha_mode_name, alpha_cmap, alpha_vmin = build_prob_viz_map(
            alpha_base[h], alpha_viz
        )
        opt_map, _alpha_mode_name2, _alpha_cmap2, _alpha_vmin2 = build_prob_viz_map(
            alpha_opt[h], alpha_viz
        )
        head_label = head_labels[h]

        diff_map = signed_log_map(alpha_opt[h] - alpha_base[h], diff_log_eps)
        diff_lim = max(diff_map.abs().max().item(), 1e-8)
        diff_v_map = signed_log_map(diff_v[h], diff_log_eps)
        diff_v_lim = max(diff_v_map.abs().max().item(), 1e-8)

        im0 = axes[h, 0].imshow(
            _to_plot_array(base_map),
            aspect="auto",
            cmap=alpha_cmap,
            vmin=alpha_vmin,
        )
        axes[h, 0].set_title(f"Head {head_label}: baseline ({alpha_mode_name})")
        fig.colorbar(im0, ax=axes[h, 0], fraction=0.046)

        im1 = axes[h, 1].imshow(
            _to_plot_array(opt_map),
            aspect="auto",
            cmap=alpha_cmap,
            vmin=alpha_vmin,
        )
        axes[h, 1].set_title(f"Head {head_label}: optimal ({alpha_mode_name})")
        fig.colorbar(im1, ax=axes[h, 1], fraction=0.046)

        im2 = axes[h, 2].imshow(
            _to_plot_array(diff_map),
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_lim,
            vmax=diff_lim,
        )
        axes[h, 2].set_title(f"Head {head_label}: diff signed-log (eps={diff_log_eps:g})")
        fig.colorbar(im2, ax=axes[h, 2], fraction=0.046)

        im3 = axes[h, 3].imshow(
            _to_plot_array(diff_v_map),
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_v_lim,
            vmax=diff_v_lim,
        )
        axes[h, 3].set_title(
            f"Head {head_label}: diff_v=delta alpha*|V| signed-log (eps={diff_log_eps:g})"
        )
        fig.colorbar(im3, ax=axes[h, 3], fraction=0.046)

        for col in range(4):
            axes[h, col].set_xlabel("Key position")
            axes[h, col].set_ylabel("Query position")

    fig.savefig(out_path, dpi=dpi)
    print(f"Saved routing grid plot: {out_path}")
    plt.close(fig)
