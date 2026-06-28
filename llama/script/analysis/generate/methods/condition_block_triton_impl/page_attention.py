"""Fused representative/selected-page decode attention kernel."""

import triton
import triton.language as tl

@triton.jit(
    do_not_specialize=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
    do_not_specialize_on_alignment=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
)
def _condition_block_finalize_attention_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    v_bar_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    selected_out_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    suffix_len,
    k_suffix_head_stride,
    k_suffix_token_stride,
    v_suffix_head_stride,
    v_suffix_token_stride,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    n_chunks,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    STORE_SELECTED: tl.constexpr,
):
    """Finalize routing and immediately consume it for hybrid attention."""
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    m_mask = m_off < group_size
    block_idx = chunk * BLOCK_N + n_off
    block_mask = block_idx < n_blocks
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off
    mask_2d = m_mask[:, None] & block_mask[None, :]

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    count = tl.load(counts_ptr + block_idx, mask=block_mask, other=0)
    active_block = block_mask & (count > 0)
    active_2d = m_mask[:, None] & active_block[None, :]
    s = tl.load(
        s_cache_ptr + row[:, None] * n_blocks + block_idx[None, :],
        mask=active_2d,
        other=0.0,
    ).to(tl.float32)
    delta = tl.load(
        delta_cache_ptr + row[:, None] * n_blocks + block_idx[None, :],
        mask=active_2d,
        other=0.0,
    ).to(tl.float32)
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    z_m = tl.load(z_m_ptr + row, mask=m_mask, other=0.0)
    z_l = tl.load(z_l_ptr + row, mask=m_mask, other=1.0)
    c_m = tl.load(c_m_ptr + row, mask=m_mask, other=0.0)
    c_l = tl.load(c_l_ptr + row, mask=m_mask, other=1.0)
    cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
    tanh_half = 2.0 / (1.0 + tl.exp(-delta)) - 1.0
    b_c = tl.load(
        v_norm_ptr + kv_head * n_blocks + block_idx,
        mask=active_block,
        other=0.0,
    ).to(tl.float32)
    b_all = tl.load(v_norm_all_ptr + kv_head).to(tl.float32)
    term1 = (
        2.0
        * b_all
        * tl.exp(z - c_m[:, None])
        * (cosh_delta - 1.0)
        / c_l[:, None]
    )
    term2 = (
        2.0
        * b_c[None, :]
        * tl.exp(z - z_m[:, None])
        * tanh_half
        / z_l[:, None]
    )
    condition = tl.where(active_2d, term1 + term2, 0.0)
    selected = ((tl.sum(condition, axis=0) / group_size) > eps) & active_block
    if STORE_SELECTED:
        tl.store(
            selected_out_ptr + row[:, None] * n_blocks + block_idx[None, :],
            selected[None, :],
            mask=mask_2d,
        )

    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Unselected blocks contribute their representative directly from the
    # selection score already resident in registers/cache.
    active_rep = active_block & (~selected)
    rep_scores = tl.where(
        m_mask[:, None] & active_rep[None, :], z, -float("inf")
    )
    has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
    rep_m = tl.max(rep_scores, axis=1)
    new_m = tl.where(has_rep & m_mask, rep_m, softmax_m)
    alpha = tl.where(has_rep & m_mask, tl.exp(softmax_m - new_m), 1.0)
    beta = tl.where(
        m_mask[:, None] & active_rep[None, :],
        tl.exp(rep_scores - new_m[:, None]),
        0.0,
    )
    rep_off = (
        (kv_head * n_blocks + block_idx[:, None]) * head_dim + d_off[None, :]
    )
    v_rep = tl.load(
        v_bar_ptr + rep_off,
        mask=active_rep[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    acc = acc * alpha[:, None] + tl.dot(
        beta.to(tl.bfloat16), v_rep, out_dtype=tl.float32
    )
    softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
    softmax_m = new_m

    # Selected pages are loaded only after routing. PAGE_SIZE is independent
    # from the 16-token MMA tile, so a 32-token page is consumed as two tiles.
    for local_block in tl.static_range(0, BLOCK_N):
        page_selected = tl.sum(
            (selected & (n_off == local_block)).to(tl.int32), axis=0
        ) > 0
        if page_selected:
            page = chunk * BLOCK_N + local_block
            page_count = tl.load(counts_ptr + page)
            for page_start in tl.static_range(0, PAGE_SIZE, BLOCK_N):
                token_idx = page_start + n_off
                token_active = token_idx < page_count
                token_off = (
                    ((kv_head * n_blocks + page) * PAGE_SIZE + token_idx[:, None])
                    * head_dim
                    + d_off[None, :]
                )
                k = tl.load(
                    k_block_ptr + token_off,
                    mask=token_active[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.bfloat16)
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
                scores = tl.where(
                    m_mask[:, None] & token_active[None, :],
                    scores,
                    -float("inf"),
                )
                tile_m = tl.max(scores, axis=1)
                new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
                alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
                beta = tl.where(
                    m_mask[:, None] & token_active[None, :],
                    tl.exp(scores - new_m[:, None]),
                    0.0,
                )
                v = tl.load(
                    v_block_ptr + token_off,
                    mask=token_active[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.bfloat16)
                acc = acc * alpha[:, None] + tl.dot(
                    beta.to(tl.bfloat16), v, out_dtype=tl.float32
                )
                softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
                softmax_m = new_m
    if chunk == n_chunks - 1:
        suffix_start = 0
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + n_off
            suffix_active = suffix_idx < suffix_len
            k_suffix_off = (
                kv_head * k_suffix_head_stride
                + suffix_idx[:, None] * k_suffix_token_stride
                + d_off[None, :]
            )
            k = tl.load(
                k_suffix_ptr + k_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                m_mask[:, None] & suffix_active[None, :], scores, -float("inf")
            )
            tile_m = tl.max(scores, axis=1)
            new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
            alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
            beta = tl.where(
                m_mask[:, None] & suffix_active[None, :],
                tl.exp(scores - new_m[:, None]),
                0.0,
            )
            v_suffix_off = (
                kv_head * v_suffix_head_stride
                + suffix_idx[:, None] * v_suffix_token_stride
                + d_off[None, :]
            )
            v = tl.load(
                v_suffix_ptr + v_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                beta.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
            softmax_m = new_m
            suffix_start += BLOCK_N

    partial_off = row * n_chunks + chunk
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )
