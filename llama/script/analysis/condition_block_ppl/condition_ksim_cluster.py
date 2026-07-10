"""Simple K-similarity clustering attention helpers.

This is the intentionally small baseline: cluster tokens by normalized K
similarity, replace each cluster by mean K/V, and attend over cluster
representatives.  There is no hierarchy, condition check, or Triton path.
"""

import math

import torch
import torch.nn.functional as F


def resolve_num_clusters(seq_len, cluster_size, max_clusters=None):
    seq_len = int(seq_len)
    cluster_size = int(cluster_size)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")
    n_clusters = max(1, math.ceil(seq_len / cluster_size))
    if max_clusters is not None:
        n_clusters = min(n_clusters, int(max_clusters))
    return min(n_clusters, seq_len)


def _assign_to_centers(x, centers, chunk_size=4096):
    assignments = []
    for start in range(0, int(x.shape[0]), int(chunk_size)):
        scores = x[start : start + chunk_size] @ centers.t()
        assignments.append(scores.argmax(dim=-1))
    return torch.cat(assignments, dim=0)


def spherical_kmeans_assign(
    k_tokens,
    *,
    cluster_size,
    kmeans_iters=4,
    max_clusters=None,
    assign_chunk_size=4096,
):
    """Return token -> cluster assignments for one KV head.

    The implementation is deterministic: centers are initialized from evenly
    spaced K vectors.  Empty centers keep their previous value.
    """

    if k_tokens.ndim != 2:
        raise ValueError("k_tokens must have shape [seq_len, head_dim]")
    seq_len = int(k_tokens.shape[0])
    n_clusters = resolve_num_clusters(seq_len, cluster_size, max_clusters=max_clusters)
    if n_clusters == seq_len:
        return torch.arange(seq_len, device=k_tokens.device, dtype=torch.long)

    x = F.normalize(k_tokens.float(), p=2, dim=-1, eps=1e-6)
    init_idx = torch.linspace(
        0,
        seq_len - 1,
        steps=n_clusters,
        device=k_tokens.device,
    ).round().long()
    centers = x[init_idx].clone()
    assignments = None
    for _ in range(max(int(kmeans_iters), 1)):
        assignments = _assign_to_centers(x, centers, chunk_size=assign_chunk_size)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(n_clusters, device=k_tokens.device, dtype=torch.float32)
        new_centers.index_add_(0, assignments, x)
        counts.index_add_(
            0,
            assignments,
            torch.ones(seq_len, device=k_tokens.device, dtype=torch.float32),
        )
        nonempty = counts > 0
        centers = torch.where(
            nonempty[:, None],
            F.normalize(
                new_centers / counts.clamp_min(1.0)[:, None],
                p=2,
                dim=-1,
                eps=1e-6,
            ),
            centers,
        )
    if assignments is None:
        assignments = _assign_to_centers(x, centers, chunk_size=assign_chunk_size)
    return assignments.long()


def batched_spherical_kmeans_assign(
    k_tokens,
    *,
    cluster_size,
    kmeans_iters=4,
    max_clusters=None,
    assign_chunk_size=4096,
):
    """Return token -> cluster assignments for multiple KV heads.

    k_tokens has shape [n_heads, seq_len, head_dim].  Each head is clustered
    independently, but the assignment and center-update matmuls are batched.
    """

    if k_tokens.ndim != 3:
        raise ValueError("k_tokens must have shape [n_heads, seq_len, head_dim]")
    n_heads, seq_len, _head_dim = k_tokens.shape
    n_heads = int(n_heads)
    seq_len = int(seq_len)
    n_clusters = resolve_num_clusters(seq_len, cluster_size, max_clusters=max_clusters)
    if n_clusters == seq_len:
        return torch.arange(seq_len, device=k_tokens.device, dtype=torch.long).expand(
            n_heads,
            -1,
        )

    x = F.normalize(k_tokens.float(), p=2, dim=-1, eps=1e-6)
    init_idx = torch.linspace(
        0,
        seq_len - 1,
        steps=n_clusters,
        device=k_tokens.device,
    ).round().long()
    centers = x[:, init_idx].clone()
    head_offsets = (
        torch.arange(n_heads, device=k_tokens.device, dtype=torch.long) * n_clusters
    ).view(n_heads, 1)
    assignments = None
    for _ in range(max(int(kmeans_iters), 1)):
        chunks = []
        for start in range(0, seq_len, int(assign_chunk_size)):
            scores = torch.einsum(
                "gtd,gcd->gtc",
                x[:, start : start + int(assign_chunk_size)],
                centers,
            )
            chunks.append(scores.argmax(dim=-1))
        assignments = torch.cat(chunks, dim=1)

        flat_assign = (assignments + head_offsets).reshape(-1)
        new_centers = torch.zeros(
            n_heads * n_clusters,
            x.shape[-1],
            device=k_tokens.device,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            n_heads * n_clusters,
            device=k_tokens.device,
            dtype=torch.float32,
        )
        new_centers.index_add_(0, flat_assign, x.reshape(-1, x.shape[-1]))
        counts.index_add_(
            0,
            flat_assign,
            torch.ones(n_heads * seq_len, device=k_tokens.device, dtype=torch.float32),
        )
        new_centers = new_centers.view(n_heads, n_clusters, -1)
        counts = counts.view(n_heads, n_clusters)
        nonempty = counts > 0
        centers = torch.where(
            nonempty[..., None],
            F.normalize(
                new_centers / counts.clamp_min(1.0)[..., None],
                p=2,
                dim=-1,
                eps=1e-6,
            ),
            centers,
        )
    if assignments is None:
        chunks = []
        for start in range(0, seq_len, int(assign_chunk_size)):
            scores = torch.einsum(
                "gtd,gcd->gtc",
                x[:, start : start + int(assign_chunk_size)],
                centers,
            )
            chunks.append(scores.argmax(dim=-1))
        assignments = torch.cat(chunks, dim=1)
    return assignments.long()


def build_ksim_prefix_tensors(
    k_head,
    v_head,
    *,
    cluster_size,
    kmeans_iters=4,
    max_clusters=None,
):
    """Build causal prefix sums for arbitrary K-sim clusters on one KV head."""

    if k_head.ndim != 2 or v_head.ndim != 2:
        raise ValueError("k_head and v_head must have shape [seq_len, head_dim]")
    if int(k_head.shape[0]) != int(v_head.shape[0]):
        raise ValueError("k_head and v_head must have the same seq_len")
    seq_len, head_dim = k_head.shape
    n_clusters = resolve_num_clusters(
        seq_len,
        cluster_size,
        max_clusters=max_clusters,
    )
    assignments = spherical_kmeans_assign(
        k_head,
        cluster_size=cluster_size,
        kmeans_iters=kmeans_iters,
        max_clusters=max_clusters,
    )
    device = k_head.device
    token_idx = torch.arange(seq_len, device=device, dtype=torch.long)
    count_updates = torch.zeros(seq_len, n_clusters, device=device, dtype=torch.float32)
    k_updates = torch.zeros(seq_len, n_clusters, head_dim, device=device, dtype=torch.float32)
    v_updates = torch.zeros(seq_len, n_clusters, head_dim, device=device, dtype=torch.float32)
    count_updates[token_idx, assignments] = 1.0
    k_updates[token_idx, assignments] = k_head.float()
    v_updates[token_idx, assignments] = v_head.float()
    counts_prefix = count_updates.cumsum(dim=0)
    k_prefix = k_updates.cumsum(dim=0)
    v_prefix = v_updates.cumsum(dim=0)
    return {
        "assignments": assignments,
        "counts_prefix": counts_prefix,
        "k_prefix": k_prefix,
        "v_prefix": v_prefix,
        "n_clusters": n_clusters,
        "cluster_size": int(cluster_size),
    }


def ksim_prefix_outputs_for_queries(q_pos, pos_tensor, prefix):
    """Compute cluster-representative attention for selected query positions.

    q_pos: [n_query_heads, n_query, head_dim]
    pos_tensor: [n_query]
    """

    n_heads, n_query, head_dim = q_pos.shape
    scale = math.sqrt(head_dim)
    counts = prefix["counts_prefix"][pos_tensor].float()
    k_sum = prefix["k_prefix"][pos_tensor]
    v_sum = prefix["v_prefix"][pos_tensor]
    cluster_exists = counts > 0
    counts_safe = counts.clamp_min(1.0)
    k_bar = k_sum / counts_safe[..., None]
    v_bar = v_sum / counts_safe[..., None]
    logits = torch.einsum("hqd,qcd->hqc", q_pos.float(), k_bar) / scale
    logits = logits + torch.log(counts_safe).view(1, n_query, -1)
    logits = logits.masked_fill(~cluster_exists.view(1, n_query, -1), float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    output = torch.einsum("hqc,qcd->hqd", weights, v_bar)
    visible_clusters = int(cluster_exists.sum().item()) * int(n_heads)
    total_available = int((pos_tensor.long() + 1).sum().item()) * int(n_heads)
    return output, {
        "rows": int(n_heads * n_query),
        "clusters": visible_clusters,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": visible_clusters,
        "total_available": total_available,
    }


def build_prompt_ksim_clusters(
    k_all,
    v_all,
    *,
    cluster_size,
    kmeans_iters=4,
    max_clusters=None,
):
    """Build prompt clusters for all KV heads.

    k_all/v_all: [n_kv_heads, prompt_len, head_dim]
    """

    k_bars = []
    v_bars = []
    counts = []
    assignments = []
    for kv_head in range(int(k_all.shape[0])):
        n_clusters = resolve_num_clusters(
            int(k_all.shape[1]),
            cluster_size,
            max_clusters=max_clusters,
        )
        assign = spherical_kmeans_assign(
            k_all[kv_head],
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
            max_clusters=max_clusters,
        )
        head_dim = int(k_all.shape[-1])
        k_sum = torch.zeros(n_clusters, head_dim, device=k_all.device, dtype=torch.float32)
        v_sum = torch.zeros(n_clusters, head_dim, device=v_all.device, dtype=torch.float32)
        count = torch.zeros(n_clusters, device=k_all.device, dtype=torch.float32)
        k_sum.index_add_(0, assign, k_all[kv_head].float())
        v_sum.index_add_(0, assign, v_all[kv_head].float())
        count.index_add_(
            0,
            assign,
            torch.ones(int(k_all.shape[1]), device=k_all.device, dtype=torch.float32),
        )
        k_bars.append(k_sum / count.clamp_min(1.0)[:, None])
        v_bars.append(v_sum / count.clamp_min(1.0)[:, None])
        counts.append(count)
        assignments.append(assign)

    return {
        "k_bar": torch.stack(k_bars, dim=0),
        "v_bar": torch.stack(v_bars, dim=0),
        "counts": torch.stack(counts, dim=0),
        "assignments": assignments,
        "cluster_size": int(cluster_size),
    }
