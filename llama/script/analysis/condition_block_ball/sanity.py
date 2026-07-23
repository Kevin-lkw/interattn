"""Soundness and parity checks for the ball-bound delta.

Checks, on synthetic structured keys (no model download needed):

1. PPL harness (`ppl_condition.ball_delta_for_queries`): ball delta >= exact
   per-token delta for every (head, query, block) pair over several key
   geometries and partial causal prefixes; report ball/box looseness vs exact.
2. Generation harness (`gen_selection`): ball delta >= dense exact delta on a
   BF16 non-divisible prompt; `z_logits`/`v_bar`/sizes bitwise equal to the box
   eager selection; eps extremes select all existing blocks / none.
"""

import argparse
import math

import torch

from ..condition_block_gen.methods.condition_block_triton_impl import core
from ..condition_block_ppl import runner_cond_block as rcb
from .gen_selection import ball_radius, select_prompt_blocks_ball
from .ppl_condition import ball_delta_for_queries, diag_ell_delta_for_queries


def _make_keys(kind, n_heads, seq_len, head_dim, device, generator):
    k = torch.randn(n_heads, seq_len, head_dim, device=device, generator=generator)
    if kind == "drift":
        # RoPE-like smooth drift along one shared direction per head.
        direction = torch.randn(n_heads, 1, head_dim, device=device, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        ramp = torch.linspace(0, 4.0, seq_len, device=device).view(1, seq_len, 1)
        k = 0.3 * k + ramp * direction
    elif kind == "outlier":
        k[:, ::37] *= 8.0
    elif kind == "real_scale":
        k = k * 3.0 + 1.5
    return k


def check_ppl_harness(device):
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    n_heads, seq_len, head_dim, block_size = 4, 512, 128, 32
    scale = math.sqrt(head_dim)
    pos_list = [17, 100, 255, 311, 511]
    pos_tensor = torch.tensor(pos_list, device=device, dtype=torch.long)
    total_pairs = 0
    min_slack = float("inf")
    for kind in ("iid", "drift", "outlier", "real_scale"):
        k = _make_keys(kind, n_heads, seq_len, head_dim, device, generator)
        v = torch.randn(n_heads, seq_len, head_dim, device=device, generator=generator)
        q = torch.randn(n_heads, len(pos_list), head_dim, device=device, generator=generator)
        prefix = rcb._build_block_prefix_tensors(k, v, block_size)
        n_blocks = prefix["block_starts"].numel()

        raw_prefix_len = pos_tensor[:, None] - prefix["block_starts"][None, :] + 1
        prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
        size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
        cluster_exists = size > 0
        prefix_idx = (size - 1).clamp_min(0)
        size_float = size.clamp_min(1).float()
        k_bar = rcb._gather_prefix(prefix["k_cumsum"], prefix_idx) / size_float.view(
            1, len(pos_list), n_blocks, 1
        )
        token_visible = (
            prefix["valid_token"][None, :, :]
            & (prefix["token_idx"][None, :, :] <= pos_tensor[:, None, None])
        )
        s_c = (q[:, :, None, :] * k_bar).sum(dim=-1) / scale
        block_logits = torch.einsum("hqd,hbtd->hqbt", q, prefix["k_block"]) / scale

        # Exact per-token delta (delta_mode == "exact" in the runner).
        centered = (block_logits - s_c.unsqueeze(-1)).abs()
        delta_exact = centered.masked_fill(~token_visible.unsqueeze(0), float("-inf")).amax(dim=-1)
        delta_exact = delta_exact.masked_fill(~cluster_exists.unsqueeze(0), 0.0)

        # Box delta (delta_mode == "range_bound" in the runner).
        k_max = rcb._gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = rcb._gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_b = q[:, :, None, :]
        upper = torch.maximum(q_b * k_max, q_b * k_min).sum(dim=-1) / scale
        lower = torch.minimum(q_b * k_max, q_b * k_min).sum(dim=-1) / scale
        delta_box = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
        delta_box = delta_box.masked_fill(~cluster_exists.unsqueeze(0), 0.0)

        delta_ball = ball_delta_for_queries(
            q_pos=q,
            k_bar=k_bar,
            prefix=prefix,
            token_visible=token_visible,
            cluster_exists=cluster_exists,
            scale=scale,
        )

        delta_diag = diag_ell_delta_for_queries(
            q_pos=q,
            k_bar=k_bar,
            prefix=prefix,
            prefix_idx=prefix_idx,
            token_visible=token_visible,
            cluster_exists=cluster_exists,
            scale=scale,
        )

        exists = cluster_exists.unsqueeze(0).expand_as(delta_exact)
        violation = (delta_ball[exists] < delta_exact[exists] - 1e-4).sum().item()
        assert violation == 0, f"{kind}: {violation} ball soundness violations"
        violation_diag = (delta_diag[exists] < delta_exact[exists] - 1e-4).sum().item()
        assert violation_diag == 0, f"{kind}: {violation_diag} diag_ell soundness violations"
        keep = exists & (delta_exact > 0.01)
        ratio_ball = (delta_ball[keep] / delta_exact[keep]).median().item()
        ratio_box = (delta_box[keep] / delta_exact[keep]).median().item()
        ratio_diag = (delta_diag[keep] / delta_exact[keep]).median().item()
        min_slack = min(min_slack, (delta_ball[keep] / delta_exact[keep]).min().item())
        total_pairs += int(exists.sum().item())
        print(
            f"[ppl:{kind}] pairs={int(exists.sum())} violations=0 "
            f"median ball/exact={ratio_ball:.2f} box/exact={ratio_box:.2f} "
            f"diag_ell/exact={ratio_diag:.2f}"
        )
    print(f"[ppl] all {total_pairs} pairs sound; min ball/exact ratio {min_slack:.3f}")


def check_gen_harness(device):
    generator = torch.Generator(device=device).manual_seed(1)
    n_kv, group, seq_len, head_dim, block_size = 2, 4, 1000, 128, 32
    scale = math.sqrt(head_dim)
    k = _make_keys("drift", n_kv, seq_len, head_dim, device, generator).to(torch.bfloat16)
    v = torch.randn(n_kv, seq_len, head_dim, device=device, generator=generator).to(torch.bfloat16)
    prefix = core._build_prompt_blocks(k, v, block_size)
    q = torch.randn(n_kv, group, 1, head_dim, device=device, generator=generator)

    ball = select_prompt_blocks_ball(q, prefix, eps=0.1)
    box = core._select_prompt_blocks_eager(q, prefix, eps=0.1)
    for name, index in (("z_logits", 1), ("v_bar", 2), ("size", 3), ("cluster_exists", 4)):
        assert torch.equal(ball[index], box[index]), f"{name} differs from box path"

    # Radius soundness against the dense exact delta on the BF16 pages.
    radius = ball_radius(prefix)
    k_pages = prefix["k_block_attn"].float()
    k_bar = prefix["k_bar"].float()
    valid = prefix["valid_token"]
    exact_dist = (k_pages - k_bar.unsqueeze(2)).norm(dim=-1)
    exact_dist = exact_dist.masked_fill(~valid.unsqueeze(0), float("-inf"))
    exact_radius = exact_dist.amax(dim=-1).clamp_min(0.0)
    assert (radius >= exact_radius - 1e-4).all(), "radius below exact max distance"

    qf = q.float()
    scores = torch.einsum("gsqd,gbtd->gsqbt", qf, k_pages) / scale
    s_c = torch.einsum("gsqd,gbd->gsqb", qf, k_bar) / scale
    exact_delta = (scores - s_c.unsqueeze(-1)).abs()
    exact_delta = exact_delta.masked_fill(~valid.view(1, 1, 1, *valid.shape), float("-inf"))
    exact_delta = exact_delta.amax(dim=-1)
    ball_delta = qf.norm(dim=-1).unsqueeze(-1) * radius[:, None, None, :] / scale
    exists = (prefix["block_valid_counts"] > 0).view(1, 1, 1, -1).expand_as(ball_delta)
    assert (ball_delta[exists] >= exact_delta[exists] - 1e-4).all(), "gen delta unsound"

    all_sel = select_prompt_blocks_ball(q, prefix, eps=0.0)[0]
    none_sel = select_prompt_blocks_ball(q, prefix, eps=1e9)[0]
    exists_mask = ball[4].view(1, 1, 1, -1).expand_as(all_sel)
    assert torch.equal(all_sel, exists_mask), "eps=0 must select every existing block"
    assert not none_sel.any(), "eps=1e9 must select nothing"
    ratio = (ball_delta[exists] / exact_delta[exists].clamp_min(1e-6)).median().item()
    print(f"[gen] parity ok, radius/delta sound, median ball/exact={ratio:.2f}, eps extremes ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    check_ppl_harness(args.device)
    check_gen_harness(args.device)
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
