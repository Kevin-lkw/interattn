"""PyTorch reference implementation of Double-P decode attention.

This module implements the algorithm described in arXiv:2602.05191 for
accuracy comparisons.  It intentionally does not reproduce the unpublished
custom GPU kernels, so its wall-clock latency is not a Double-P speed result.

The prompt middle is clustered independently for every KV head with standard
Euclidean k-means.  During decode, size-weighted centroid scores define a
cluster distribution.  Stage 1 keeps the smallest top-p1 cluster prefix;
stage 2 expands the smallest top-p2 prefix to exact tokens.  The remaining
stage-1 clusters use one mean-K/mean-V representative, and clusters outside
stage 1 are pruned.

As in the paper's Llama-3.1 experiments, sink tokens and a prompt-tail window
are always exact.  Generated tokens are also kept exact in this cached
reference path.  The latter is conservative and matches the other cached
baselines in this repository; the measured equivalent budget includes them.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from .condition_block import (
    _full_generation_step_metadata,
    _should_stop,
    _stop_token_ids,
)
from .patching import merge_stats, summarize_stats


__all__ = [
    "DoublePDecodeRunner",
    "batched_euclidean_kmeans_assign",
    "build_double_p_prompt_clusters",
    "double_p_attention",
    "double_p_decode_context",
    "full_attention_sdpa_context",
    "generate_double_p_cached",
    "summarize_double_p_step_metadata",
    "top_p_mask",
]


def _resolve_num_clusters(num_tokens: int, cluster_size: int) -> int:
    num_tokens = int(num_tokens)
    cluster_size = int(cluster_size)
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    if cluster_size <= 0:
        raise ValueError("cluster_size must be > 0")
    if num_tokens == 0:
        return 0
    return min(num_tokens, max(1, math.ceil(num_tokens / cluster_size)))


def batched_euclidean_kmeans_assign(
    k_tokens: torch.Tensor,
    *,
    cluster_size: int,
    kmeans_iters: int = 4,
    assign_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return token-to-cluster assignments for each KV head.

    ``k_tokens`` has shape ``[n_kv_heads, num_tokens, head_dim]``.  Centers
    are initialized deterministically from evenly spaced tokens.  Empty
    centers retain their previous value.
    """

    if k_tokens.ndim != 3:
        raise ValueError("k_tokens must have shape [n_kv_heads, num_tokens, head_dim]")
    if int(kmeans_iters) <= 0:
        raise ValueError("kmeans_iters must be > 0")
    if int(assign_chunk_size) <= 0:
        raise ValueError("assign_chunk_size must be > 0")

    n_kv_heads, num_tokens, head_dim = map(int, k_tokens.shape)
    n_clusters = _resolve_num_clusters(num_tokens, cluster_size)
    if num_tokens == 0:
        return torch.empty(
            n_kv_heads,
            0,
            device=k_tokens.device,
            dtype=torch.long,
        )
    if n_clusters == num_tokens:
        return torch.arange(
            num_tokens,
            device=k_tokens.device,
            dtype=torch.long,
        ).expand(n_kv_heads, -1)

    x = k_tokens.float()
    init_idx = torch.linspace(
        0,
        num_tokens - 1,
        steps=n_clusters,
        device=k_tokens.device,
    ).round().long()
    centers = x[:, init_idx].clone()
    head_offsets = (
        torch.arange(n_kv_heads, device=k_tokens.device, dtype=torch.long) * n_clusters
    ).view(n_kv_heads, 1)

    assignments = None
    for _ in range(int(kmeans_iters)):
        center_norm = centers.square().sum(dim=-1).unsqueeze(1)
        chunks = []
        for start in range(0, num_tokens, int(assign_chunk_size)):
            token_chunk = x[:, start : start + int(assign_chunk_size)]
            # argmin ||x-c||^2 == argmax 2 x.c - ||c||^2.
            scores = 2.0 * torch.einsum("gtd,gcd->gtc", token_chunk, centers)
            scores = scores - center_norm
            chunks.append(scores.argmax(dim=-1))
        assignments = torch.cat(chunks, dim=1)

        flat_assignments = (assignments + head_offsets).reshape(-1)
        center_sums = torch.zeros(
            n_kv_heads * n_clusters,
            head_dim,
            device=k_tokens.device,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            n_kv_heads * n_clusters,
            device=k_tokens.device,
            dtype=torch.float32,
        )
        center_sums.index_add_(0, flat_assignments, x.reshape(-1, head_dim))
        counts.index_add_(
            0,
            flat_assignments,
            torch.ones(n_kv_heads * num_tokens, device=k_tokens.device),
        )
        center_sums = center_sums.view(n_kv_heads, n_clusters, head_dim)
        counts = counts.view(n_kv_heads, n_clusters)
        nonempty = counts > 0
        centers = torch.where(
            nonempty.unsqueeze(-1),
            center_sums / counts.clamp_min(1.0).unsqueeze(-1),
            centers,
        )

    return assignments.long()


def build_double_p_prompt_clusters(
    k_prompt: torch.Tensor,
    v_prompt: torch.Tensor,
    *,
    cluster_size: int,
    kmeans_iters: int,
    sink_tokens: int,
    window_size: int,
) -> dict:
    """Cluster the sparse middle of a prompt and build Double-P summaries."""

    if k_prompt.ndim != 3 or v_prompt.ndim != 3:
        raise ValueError("k_prompt and v_prompt must have shape [n_kv_heads, prompt_len, head_dim]")
    if k_prompt.shape != v_prompt.shape:
        raise ValueError("k_prompt and v_prompt must have identical shapes")
    if int(sink_tokens) < 0 or int(window_size) < 0:
        raise ValueError("sink_tokens and window_size must be >= 0")

    n_kv_heads, prompt_len, head_dim = map(int, k_prompt.shape)
    middle_start = min(int(sink_tokens), prompt_len)
    middle_end = max(middle_start, prompt_len - int(window_size))
    middle_k = k_prompt[:, middle_start:middle_end]
    middle_v = v_prompt[:, middle_start:middle_end]
    num_middle_tokens = int(middle_k.shape[1])
    n_clusters = _resolve_num_clusters(num_middle_tokens, cluster_size)

    if n_clusters == 0:
        empty_summary = torch.empty(
            n_kv_heads,
            0,
            head_dim,
            device=k_prompt.device,
            dtype=torch.float32,
        )
        return {
            "assignments": torch.empty(
                n_kv_heads,
                0,
                device=k_prompt.device,
                dtype=torch.long,
            ),
            "counts": torch.empty(
                n_kv_heads,
                0,
                device=k_prompt.device,
                dtype=torch.float32,
            ),
            "k_bar": empty_summary,
            "v_bar": empty_summary.clone(),
            "middle_start": middle_start,
            "middle_end": middle_end,
            "prompt_len": prompt_len,
            "cluster_size": int(cluster_size),
        }

    assignments = batched_euclidean_kmeans_assign(
        middle_k,
        cluster_size=cluster_size,
        kmeans_iters=kmeans_iters,
    )
    head_offsets = (
        torch.arange(n_kv_heads, device=k_prompt.device, dtype=torch.long) * n_clusters
    ).view(n_kv_heads, 1)
    flat_assignments = (assignments + head_offsets).reshape(-1)
    counts = torch.zeros(
        n_kv_heads * n_clusters,
        device=k_prompt.device,
        dtype=torch.float32,
    )
    k_sum = torch.zeros(
        n_kv_heads * n_clusters,
        head_dim,
        device=k_prompt.device,
        dtype=torch.float32,
    )
    v_sum = torch.zeros_like(k_sum)
    counts.index_add_(
        0,
        flat_assignments,
        torch.ones(n_kv_heads * num_middle_tokens, device=k_prompt.device),
    )
    k_sum.index_add_(0, flat_assignments, middle_k.float().reshape(-1, head_dim))
    v_sum.index_add_(0, flat_assignments, middle_v.float().reshape(-1, head_dim))
    counts = counts.view(n_kv_heads, n_clusters)
    k_sum = k_sum.view(n_kv_heads, n_clusters, head_dim)
    v_sum = v_sum.view(n_kv_heads, n_clusters, head_dim)

    return {
        "assignments": assignments,
        "counts": counts,
        "k_bar": k_sum / counts.clamp_min(1.0).unsqueeze(-1),
        "v_bar": v_sum / counts.clamp_min(1.0).unsqueeze(-1),
        "middle_start": middle_start,
        "middle_end": middle_end,
        "prompt_len": prompt_len,
        "cluster_size": int(cluster_size),
    }


def top_p_mask(probabilities: torch.Tensor, p: float) -> torch.Tensor:
    """Select the minimal descending-probability prefix with mass >= ``p``."""

    p = float(p)
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if probabilities.ndim < 1:
        raise ValueError("probabilities must have at least one dimension")
    if int(probabilities.shape[-1]) == 0:
        return torch.zeros_like(probabilities, dtype=torch.bool)
    if p == 1.0:
        # The caller intersects this with its active-cluster mask.  Selecting
        # every slot avoids dropping a finite but softmax-underflowed cluster
        # at the dense-equivalence validation point.
        return torch.ones_like(probabilities, dtype=torch.bool)

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        dim=-1,
        descending=True,
    )
    cumulative_before = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
    sorted_selected = cumulative_before < p
    selected = torch.zeros_like(sorted_selected)
    selected.scatter_(-1, sorted_indices, sorted_selected)
    return selected


def double_p_attention(
    *,
    q_grouped: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    prompt_clusters: dict,
    pos_tensor: torch.Tensor,
    p1: float,
    p2: float,
) -> tuple[torch.Tensor, dict]:
    """Compute mixed exact/centroid Double-P attention for cached decode."""

    if not 0 < float(p2) <= float(p1) <= 1:
        raise ValueError("Double-P requires 0 < p2 <= p1 <= 1")
    n_kv_heads, group_size, n_query, head_dim = map(int, q_grouped.shape)
    if k_all.shape[:1] != (n_kv_heads,) or v_all.shape != k_all.shape:
        raise ValueError("k_all/v_all must match the KV-head dimension of q_grouped")
    if int(n_query) != int(pos_tensor.numel()):
        raise ValueError("pos_tensor must contain one position per query")

    scale = math.sqrt(head_dim)
    q_float = q_grouped.float()
    middle_start = int(prompt_clusters["middle_start"])
    middle_end = int(prompt_clusters["middle_end"])
    prompt_len = int(prompt_clusters["prompt_len"])
    middle_k = k_all[:, middle_start:middle_end]
    middle_v = v_all[:, middle_start:middle_end]
    counts = prompt_clusters["counts"]
    n_clusters = int(counts.shape[-1])

    cluster_exists = counts > 0
    if n_clusters > 0:
        cluster_logits = torch.einsum(
            "grqd,gcd->grqc",
            q_float,
            prompt_clusters["k_bar"],
        ) / scale
        cluster_logits = cluster_logits + torch.log(counts.clamp_min(1.0)).view(
            n_kv_heads,
            1,
            1,
            n_clusters,
        )
        cluster_active = cluster_exists.view(n_kv_heads, 1, 1, n_clusters).expand_as(
            cluster_logits
        )
        cluster_logits = cluster_logits.masked_fill(~cluster_active, float("-inf"))
        cluster_probabilities = torch.softmax(cluster_logits, dim=-1).masked_fill(
            ~cluster_active,
            0.0,
        )
        stage1_mask = top_p_mask(cluster_probabilities, p1) & cluster_active
        exact_cluster_mask = top_p_mask(cluster_probabilities, p2) & stage1_mask
        approx_cluster_mask = stage1_mask & ~exact_cluster_mask
        pruned_cluster_mask = cluster_active & ~stage1_mask

        assignments = prompt_clusters["assignments"]
        token_cluster = assignments.view(n_kv_heads, 1, 1, -1).expand(
            n_kv_heads,
            group_size,
            n_query,
            -1,
        )
        exact_middle_mask = torch.gather(
            exact_cluster_mask,
            dim=-1,
            index=token_cluster,
        )
        exact_middle_logits = torch.einsum("grqd,gtd->grqt", q_float, middle_k.float()) / scale
        exact_middle_logits = exact_middle_logits.masked_fill(
            ~exact_middle_mask,
            float("-inf"),
        )
        approx_cluster_logits = cluster_logits.masked_fill(
            ~approx_cluster_mask,
            float("-inf"),
        )
    else:
        cluster_active = torch.zeros(
            n_kv_heads,
            group_size,
            n_query,
            0,
            device=q_grouped.device,
            dtype=torch.bool,
        )
        stage1_mask = cluster_active
        exact_cluster_mask = cluster_active
        approx_cluster_mask = cluster_active
        pruned_cluster_mask = cluster_active
        exact_middle_mask = torch.zeros(
            n_kv_heads,
            group_size,
            n_query,
            0,
            device=q_grouped.device,
            dtype=torch.bool,
        )
        exact_middle_logits = None
        approx_cluster_logits = None

    exact_ranges = (
        torch.arange(middle_start, device=k_all.device),
        torch.arange(middle_end, prompt_len, device=k_all.device),
        torch.arange(prompt_len, int(k_all.shape[1]), device=k_all.device),
    )
    exact_positions = torch.cat(exact_ranges, dim=0)
    exact_k = torch.cat(
        [
            k_all[:, :middle_start],
            k_all[:, middle_end:prompt_len],
            k_all[:, prompt_len:],
        ],
        dim=1,
    )
    exact_v = torch.cat(
        [
            v_all[:, :middle_start],
            v_all[:, middle_end:prompt_len],
            v_all[:, prompt_len:],
        ],
        dim=1,
    )
    exact_other_logits = None
    exact_other_mask = None
    if int(exact_k.shape[1]) > 0:
        exact_other_logits = torch.einsum("grqd,gtd->grqt", q_float, exact_k.float()) / scale
        # Cached generation has one query and no future suffix keys.  The PPL
        # runner evaluates several teacher-forced decode positions together,
        # so explicitly mask future continuation tokens here.
        exact_other_mask = exact_positions.view(1, 1, 1, -1) <= pos_tensor.view(
            1,
            1,
            n_query,
            1,
        )
        exact_other_logits = exact_other_logits.masked_fill(
            ~exact_other_mask,
            float("-inf"),
        )

    max_parts = []
    if exact_middle_logits is not None and exact_middle_mask.any():
        max_parts.append(exact_middle_logits.amax(dim=-1))
    if approx_cluster_logits is not None and approx_cluster_mask.any():
        max_parts.append(approx_cluster_logits.amax(dim=-1))
    if exact_other_logits is not None:
        max_parts.append(exact_other_logits.amax(dim=-1))
    if not max_parts:
        raise RuntimeError(
            "Double-P decode produced no active exact tokens or cluster representatives"
        )

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    normalizer = torch.zeros_like(max_logit)
    numerator = torch.zeros(
        *max_logit.shape,
        head_dim,
        device=q_grouped.device,
        dtype=torch.float32,
    )

    if exact_middle_logits is not None and exact_middle_mask.any():
        middle_exp = torch.exp(exact_middle_logits - max_logit.unsqueeze(-1)).masked_fill(
            ~exact_middle_mask,
            0.0,
        )
        normalizer = normalizer + middle_exp.sum(dim=-1)
        numerator = numerator + torch.einsum(
            "grqt,gtd->grqd",
            middle_exp,
            middle_v.float(),
        )
    if approx_cluster_logits is not None and approx_cluster_mask.any():
        cluster_exp = torch.exp(
            approx_cluster_logits - max_logit.unsqueeze(-1)
        ).masked_fill(~approx_cluster_mask, 0.0)
        normalizer = normalizer + cluster_exp.sum(dim=-1)
        numerator = numerator + torch.einsum(
            "grqc,gcd->grqd",
            cluster_exp,
            prompt_clusters["v_bar"],
        )
    if exact_other_logits is not None:
        other_exp = torch.exp(exact_other_logits - max_logit.unsqueeze(-1)).masked_fill(
            ~exact_other_mask,
            0.0,
        )
        normalizer = normalizer + other_exp.sum(dim=-1)
        numerator = numerator + torch.einsum(
            "grqt,gtd->grqd",
            other_exp,
            exact_v.float(),
        )

    counts_per_row = counts.view(n_kv_heads, 1, 1, n_clusters)
    expanded_middle_tokens = int(
        (exact_cluster_mask.long() * counts_per_row.long()).sum().item()
    )
    exact_other_tokens = (
        int(exact_other_mask.sum().item()) * n_kv_heads * group_size
        if exact_other_mask is not None
        else 0
    )
    selected_tokens = expanded_middle_tokens + exact_other_tokens
    approx_clusters = int(approx_cluster_mask.sum().item())
    total_available = int((pos_tensor.long() + 1).sum().item()) * n_kv_heads * group_size
    stats = {
        "rows": n_kv_heads * group_size * n_query,
        "clusters": int(cluster_active.sum().item()),
        "stage1_clusters": int(stage1_mask.sum().item()),
        "selected_clusters": int(exact_cluster_mask.sum().item()),
        "approx_clusters": approx_clusters,
        "pruned_clusters": int(pruned_cluster_mask.sum().item()),
        "selected_tokens": selected_tokens,
        "exact_other_tokens": exact_other_tokens,
        "hybrid_tokens": selected_tokens + approx_clusters,
        "total_available": total_available,
    }
    output = numerator / normalizer.clamp_min(1e-30).unsqueeze(-1)
    return output, stats


def _sdpa_full_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **_kwargs,
):
    dropout_p = float(dropout) if module.training else 0.0
    is_full_sequence_causal = query.shape[2] == key.shape[2]
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None if is_full_sequence_causal else attention_mask,
        dropout_p=dropout_p,
        is_causal=is_full_sequence_causal,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _full_attention_stats_for_heads(n_heads: int, pos: int) -> dict:
    total_available = (int(pos) + 1) * int(n_heads)
    return {
        "rows": int(n_heads),
        "clusters": 0,
        "stage1_clusters": 0,
        "selected_clusters": 0,
        "approx_clusters": 0,
        "pruned_clusters": 0,
        "selected_tokens": total_available,
        "exact_other_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


class DoublePDecodeRunner:
    def __init__(
        self,
        *,
        model,
        full_attention_layers: int,
        cluster_size: int,
        kmeans_iters: int,
        p1: float,
        p2: float,
        sink_tokens: int,
        window_size: int,
        prompt_len: int,
        pos: int,
        prompt_cluster_cache: dict,
    ):
        self.module_to_layer_idx = {
            id(layer.self_attn): int(layer_idx)
            for layer_idx, layer in enumerate(model.model.layers)
        }
        self.full_attention_layers = int(full_attention_layers)
        self.cluster_size = int(cluster_size)
        self.kmeans_iters = int(kmeans_iters)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.sink_tokens = int(sink_tokens)
        self.window_size = int(window_size)
        self.prompt_len = int(prompt_len)
        self.pos = int(pos)
        self.prompt_cluster_cache = prompt_cluster_cache
        self.stats_by_layer = {}
        self.aggregate_stats = {}

    def should_compress(self, layer_idx: int) -> bool:
        return int(layer_idx) >= self.full_attention_layers

    def summarize(self) -> dict:
        return summarize_stats(self.aggregate_stats, self.stats_by_layer)

    def attention_forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            layer_idx = self.module_to_layer_idx.get(id(module))
        if layer_idx is None:
            return _sdpa_full_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )

        if not self.should_compress(layer_idx):
            stats = _full_attention_stats_for_heads(query.shape[1], self.pos)
            self.stats_by_layer[int(layer_idx)] = stats
            merge_stats(self.aggregate_stats, stats)
            return _sdpa_full_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )

        if query.shape[0] != 1 or query.shape[2] != 1:
            raise ValueError("double_p expects batch_size=1 and q_len=1 during decode")
        if self.pos < self.prompt_len:
            raise ValueError("double_p expects generated-token decode positions")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("double_p expects grouped-query or multi-head attention")

        q_grouped = query[0].reshape(
            n_kv_heads,
            n_heads // n_kv_heads,
            q_len,
            head_dim,
        )
        k_all = key[0]
        v_all = value[0]
        cache_key = (
            int(layer_idx),
            self.prompt_len,
            self.cluster_size,
            self.kmeans_iters,
            self.sink_tokens,
            self.window_size,
        )
        prompt_clusters = self.prompt_cluster_cache.get(cache_key)
        if prompt_clusters is None:
            prompt_clusters = build_double_p_prompt_clusters(
                k_all[:, : self.prompt_len],
                v_all[:, : self.prompt_len],
                cluster_size=self.cluster_size,
                kmeans_iters=self.kmeans_iters,
                sink_tokens=self.sink_tokens,
                window_size=self.window_size,
            )
            self.prompt_cluster_cache[cache_key] = prompt_clusters

        output, stats = double_p_attention(
            q_grouped=q_grouped,
            k_all=k_all,
            v_all=v_all,
            prompt_clusters=prompt_clusters,
            pos_tensor=torch.tensor([self.pos], device=query.device, dtype=torch.long),
            p1=self.p1,
            p2=self.p2,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)
        self.stats_by_layer[int(layer_idx)] = stats
        merge_stats(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def double_p_decode_context(runner: DoublePDecodeRunner):
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = runner.attention_forward
    try:
        yield runner
    finally:
        modeling_llama.eager_attention_forward = original_eager


@contextlib.contextmanager
def full_attention_sdpa_context():
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = _sdpa_full_attention_forward
    try:
        yield
    finally:
        modeling_llama.eager_attention_forward = original_eager


def summarize_double_p_step_metadata(step_metadata: list[dict]) -> dict:
    aggregate = {}
    by_step = []
    for step_idx, metadata in enumerate(step_metadata):
        if not metadata:
            continue
        step_aggregate = metadata.get("aggregate", {})
        by_step.append(
            {
                "step": step_idx,
                "equiv_budget": step_aggregate.get("mean_budget_causal"),
                "stage1_clusters": step_aggregate.get("stage1_clusters"),
                "exact_clusters": step_aggregate.get("selected_clusters"),
                "approx_clusters": step_aggregate.get("approx_clusters"),
                "pruned_clusters": step_aggregate.get("pruned_clusters"),
            }
        )
        for key, value in step_aggregate.items():
            if isinstance(value, int):
                aggregate[key] = int(aggregate.get(key, 0)) + int(value)

    total_available = max(int(aggregate.get("total_available", 0)), 1)
    rows = max(int(aggregate.get("rows", 0)), 1)
    hybrid_tokens = int(aggregate.get("hybrid_tokens", 0))
    equiv_budget = float(hybrid_tokens / total_available)
    return {
        "double_p_equiv_budget": equiv_budget,
        "double_p_budget": {
            **aggregate,
            "mean_hybrid_tokens": float(hybrid_tokens / rows),
            "mean_budget_causal": equiv_budget,
            "mean_budget_visible": equiv_budget,
            "by_step": by_step,
        },
    }


def generate_double_p_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    device,
    dataset=None,
):
    """Generate with dense prefill and PyTorch Double-P cached decode."""

    del device
    if int(input_ids.shape[0]) != 1:
        raise ValueError("double_p generation currently expects batch_size=1")

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(method.max_new_tokens)
    if max_new_tokens <= 0:
        output_ids = torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        return output_ids, summarize_double_p_step_metadata([])

    stop_token_ids = _stop_token_ids(tokenizer, dataset)
    prompt_cluster_cache = {}
    generated = []
    step_metadata = []
    cur_mask = attention_mask
    total_len = prompt_len

    with full_attention_sdpa_context():
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=cur_mask,
                use_cache=True,
                logits_to_keep=1,
            )

    logits = outputs.logits.float()
    past_key_values = outputs.past_key_values
    step_metadata.append(_full_generation_step_metadata(model, [total_len - 1]))

    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_id)
    if _should_stop(next_id, stop_token_ids):
        metadata = summarize_double_p_step_metadata(step_metadata)
        metadata["double_p"] = _double_p_config_metadata(method)
        return torch.cat(generated, dim=1), metadata

    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
    step_input_ids = next_id
    total_len += 1

    for _step in range(1, max_new_tokens):
        runner = DoublePDecodeRunner(
            model=model,
            full_attention_layers=method.full_attention_layers,
            cluster_size=method.double_p_cluster_size,
            kmeans_iters=method.double_p_kmeans_iters,
            p1=method.double_p_p1,
            p2=method.double_p_p2,
            sink_tokens=method.double_p_sink_tokens,
            window_size=method.double_p_window_size,
            prompt_len=prompt_len,
            pos=total_len - 1,
            prompt_cluster_cache=prompt_cluster_cache,
        )
        with double_p_decode_context(runner):
            with torch.no_grad():
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )

        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values
        step_metadata.append(runner.summarize())

        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        if _should_stop(next_id, stop_token_ids):
            break

        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        step_input_ids = next_id
        total_len += 1

    metadata = summarize_double_p_step_metadata(step_metadata)
    metadata["double_p"] = _double_p_config_metadata(method)
    return torch.cat(generated, dim=1), metadata


def _double_p_config_metadata(method) -> dict:
    return {
        "implementation": "pytorch_reference",
        "optimized_kernel": False,
        "cluster_size": int(method.double_p_cluster_size),
        "kmeans_iters": int(method.double_p_kmeans_iters),
        "p1": float(method.double_p_p1),
        "p2": float(method.double_p_p2),
        "sink_tokens": int(method.double_p_sink_tokens),
        "window_size": int(method.double_p_window_size),
        "full_attention_layers": int(method.full_attention_layers),
        "generated_tokens_exact": True,
    }
