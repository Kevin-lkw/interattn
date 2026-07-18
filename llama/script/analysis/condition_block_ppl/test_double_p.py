import argparse
import math

import pytest
import torch

from ..condition_block_gen.methods.double_p import (
    build_double_p_prompt_clusters,
    double_p_attention,
    top_p_mask,
)
from .double_p_config import (
    double_p_setting_key,
    p2_by_unique_p1,
    parse_p_setting,
)
from .multisample.run_double_p import _combined_full_chunk_budget
from .runner_double_p_full_causal import (
    causal_clustered_end,
    full_causal_double_p_attention,
)


def test_top_p_mask_selects_minimal_prefix():
    probabilities = torch.tensor([[0.50, 0.30, 0.20]])
    assert top_p_mask(probabilities, 0.70).tolist() == [[True, True, False]]
    assert top_p_mask(probabilities, 1.0).tolist() == [[True, True, True]]


def test_threshold_config_parsing_and_stable_key():
    assert parse_p_setting("0.95:0.70") == (0.95, 0.70)
    assert parse_p_setting("0.9,0.5") == (0.9, 0.5)
    assert double_p_setting_key(0.90, 0.50) == "p1=0.9_p2=0.5"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_p_setting("0.5:0.9")


def test_ppl_setting_grid_rejects_duplicate_p1_values():
    assert p2_by_unique_p1([(0.9, 0.5), (0.95, 0.7)]) == {
        0.9: 0.5,
        0.95: 0.7,
    }
    with pytest.raises(ValueError, match="unique p1"):
        p2_by_unique_p1([(0.9, 0.5), (0.9, 0.6)])
    with pytest.raises(ValueError, match="p2 <= p1"):
        p2_by_unique_p1([(0.5, 0.9)])


def test_prompt_cluster_summaries_preserve_counts_and_means():
    torch.manual_seed(3)
    k_prompt = torch.randn(2, 11, 6)
    v_prompt = torch.randn(2, 11, 6)
    clusters = build_double_p_prompt_clusters(
        k_prompt,
        v_prompt,
        cluster_size=3,
        kmeans_iters=3,
        sink_tokens=1,
        window_size=2,
    )

    middle_k = k_prompt[:, 1:9]
    middle_v = v_prompt[:, 1:9]
    assert torch.equal(
        clusters["counts"].sum(dim=-1),
        torch.full((2,), 8.0),
    )
    assert torch.allclose(
        (clusters["k_bar"] * clusters["counts"].unsqueeze(-1)).sum(dim=1),
        middle_k.sum(dim=1),
        atol=1e-6,
    )
    assert torch.allclose(
        (clusters["v_bar"] * clusters["counts"].unsqueeze(-1)).sum(dim=1),
        middle_v.sum(dim=1),
        atol=1e-6,
    )


def test_dense_thresholds_match_causal_gqa_attention():
    torch.manual_seed(7)
    n_kv_heads = 2
    group_size = 3
    n_query = 3
    head_dim = 8
    prompt_len = 9
    total_len = 12
    positions = torch.tensor([9, 10, 11])
    q_grouped = torch.randn(n_kv_heads, group_size, n_query, head_dim)
    k_all = torch.randn(n_kv_heads, total_len, head_dim)
    v_all = torch.randn(n_kv_heads, total_len, head_dim)
    clusters = build_double_p_prompt_clusters(
        k_all[:, :prompt_len],
        v_all[:, :prompt_len],
        cluster_size=2,
        kmeans_iters=3,
        sink_tokens=1,
        window_size=2,
    )

    output, stats = double_p_attention(
        q_grouped=q_grouped,
        k_all=k_all,
        v_all=v_all,
        prompt_clusters=clusters,
        pos_tensor=positions,
        p1=1.0,
        p2=1.0,
    )
    reference = torch.empty_like(output)
    for query_idx, position in enumerate(positions.tolist()):
        logits = torch.einsum(
            "grd,gtd->grt",
            q_grouped[:, :, query_idx],
            k_all[:, : position + 1],
        ) / math.sqrt(head_dim)
        weights = torch.softmax(logits, dim=-1)
        reference[:, :, query_idx] = torch.einsum(
            "grt,gtd->grd",
            weights,
            v_all[:, : position + 1],
        )

    assert torch.allclose(output, reference, atol=2e-6, rtol=2e-6)
    assert stats["hybrid_tokens"] == stats["total_available"]


def test_aligned_budget_combines_dense_prefix_and_sparse_tail():
    # Prefix positions 0 and 1 cost (1 + 2) * 2 heads * 3 layers = 18.
    tail_budget = {
        "aggregate": {
            "hybrid_tokens": 10,
            "total_available": 20,
        }
    }
    combined = _combined_full_chunk_budget(
        num_heads=2,
        num_layers=3,
        prefix_positions=[0, 1],
        tail_budget=tail_budget,
    )
    assert math.isclose(combined, 28 / 38)


def test_full_causal_cluster_prefix_never_reaches_query_or_window():
    for position in range(40):
        clustered_end = causal_clustered_end(
            position,
            cluster_size=4,
            sink_tokens=2,
            window_size=3,
        )
        if clustered_end:
            assert clustered_end <= position + 1 - 3
            assert (clustered_end - 2) % 4 == 0


def test_full_causal_dense_thresholds_match_every_causal_position():
    torch.manual_seed(11)
    n_kv_heads = 2
    group_size = 3
    seq_len = 17
    head_dim = 8
    q_all = torch.randn(n_kv_heads * group_size, seq_len, head_dim)
    k_all = torch.randn(n_kv_heads, seq_len, head_dim)
    v_all = torch.randn(n_kv_heads, seq_len, head_dim)
    positions = list(range(seq_len))

    output, stats = full_causal_double_p_attention(
        q_all=q_all,
        k_all=k_all,
        v_all=v_all,
        pos_list=positions,
        cluster_size=3,
        kmeans_iters=3,
        p1=1.0,
        p2=1.0,
        sink_tokens=1,
        window_size=2,
    )
    reference = torch.empty_like(output)
    q_grouped = q_all.reshape(
        n_kv_heads,
        group_size,
        seq_len,
        head_dim,
    )
    for position in positions:
        logits = torch.einsum(
            "grd,gtd->grt",
            q_grouped[:, :, position],
            k_all[:, : position + 1],
        ) / math.sqrt(head_dim)
        weights = torch.softmax(logits, dim=-1)
        reference[:, position] = torch.einsum(
            "grt,gtd->grd",
            weights,
            v_all[:, : position + 1],
        ).reshape(-1, head_dim)

    assert torch.allclose(output, reference, atol=3e-6, rtol=3e-6)
    assert stats["hybrid_tokens"] == stats["total_available"]
