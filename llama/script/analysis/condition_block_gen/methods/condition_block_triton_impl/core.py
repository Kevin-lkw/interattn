import contextlib
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from transformers import StaticCache
from transformers.models.llama import modeling_llama
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from ..patching import merge_stats, summarize_stats
from .legacy import (
    _condition_block_decode_output_compact_sdpa,
    _condition_block_decode_output_dense,
    _condition_block_decode_output_triton,
    _condition_block_stage2_reduce_kernel,
)
from .page_attention import _condition_block_finalize_attention_kernel
from .selection_tma import condition_block_selection_stats_tma_kernel

# Launch configuration for the decode kernels. The defaults were picked by a
# cold-L2 sweep on RTX PRO 6000 Blackwell (see README); the env overrides exist
# so the sweep can be re-run on other hardware without editing code.
_SELECT_CHUNK = int(os.environ.get("CONDITION_BLOCK_SELECT_CHUNK", "16"))
_SELECT_WARPS = os.environ.get("CONDITION_BLOCK_SELECT_WARPS")
_FINALIZE_CHUNK = int(os.environ.get("CONDITION_BLOCK_FINALIZE_CHUNK", "32"))
_FINALIZE_WARPS = int(os.environ.get("CONDITION_BLOCK_FINALIZE_WARPS", "4"))
_FUSED_PAGE_SIZES = (16, 32, 64)


def _select_warps(n_blocks):
    # The sweep found the optimum inverts with context: many warps win while
    # the summary read set is small, few warps win once it is HBM-bound.
    if _SELECT_WARPS:
        return int(_SELECT_WARPS)
    return 8 if n_blocks <= 1024 else 2


def run_prefill_only_condition_block(**_kwargs):
    raise RuntimeError(
        "condition_block now uses the cached generate path only. "
        "Call generate_condition_block_cached through generate_with_method()."
    )


def build_condition_args(method, prompt_len):
    from types import SimpleNamespace

    return SimpleNamespace(
        seq_len=int(prompt_len),
        block_size=method.condition_block_size,
        full_attention_layers=int(method.full_attention_layers),
        delta_mode=method.condition_delta_mode,
    )


def generate_condition_block_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    device,
    dataset=None,
    term1_mass_exp=False,
):
    if method.condition_delta_mode != "range_bound":
        raise ValueError("condition_block only supports delta_mode='range_bound'.")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("condition_block generate currently expects batch_size=1.")
    if os.environ.get("CONDITION_BLOCK_CUDA_GRAPH") == "1":
        return _generate_condition_block_cuda_graph(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            method=method,
            dataset=dataset,
            term1_mass_exp=term1_mass_exp,
        )

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(method.max_new_tokens)
    if max_new_tokens <= 0:
        output_ids = torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        return output_ids, summarize_condition_block_step_metadata([])

    stop_token_ids = _stop_token_ids(tokenizer, dataset)
    stop_ids_tensor = (
        torch.tensor(stop_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        if stop_token_ids
        else None
    )
    layer_idx_list = list(range(int(model.config.num_hidden_layers)))
    prompt_prefix_cache = {}
    generated = []
    step_metadata = []
    collect_stats = os.environ.get("CONDITION_BLOCK_SKIP_STATS") != "1"
    cur_mask = attention_mask
    total_len = prompt_len
    past_key_values = None
    if os.environ.get("CONDITION_BLOCK_STATIC_CACHE") == "1":
        past_key_values = StaticCache(
            config=model.config,
            max_cache_len=prompt_len + max_new_tokens,
        )

    # Use Transformers' native SDPA interface for the long prompt. This keeps
    # prefill identical to the optimized full-attention baseline; the custom
    # eager hook is installed only for single-token decode below.
    with model_attention_implementation(model, "sdpa"):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )

    logits = outputs.logits.float()
    past_key_values = outputs.past_key_values
    if os.environ.get("CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE") == "1":
        past_key_values = _static_cache_from_prefill(
            model.config,
            past_key_values,
            max_cache_len=prompt_len + max_new_tokens,
        )
    if collect_stats:
        step_metadata.append(_full_generation_step_metadata(model, [total_len - 1]))

    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_id)
    if _should_stop(next_id, stop_ids_tensor):
        return torch.cat(generated, dim=1), summarize_condition_block_step_metadata(step_metadata)

    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
    step_input_ids = next_id
    total_len += 1

    runner = ConditionBlockDecodeRunner(
        model=model,
        model_config=model.config,
        layer_idx_list=layer_idx_list,
        full_attention_layers=method.full_attention_layers,
        block_size=method.condition_block_size,
        eps=method.condition_eps,
        prompt_len=prompt_len,
        pos=total_len - 1,
        prompt_prefix_cache=prompt_prefix_cache,
        term1_mass_exp=term1_mass_exp,
    )
    with condition_block_decode_context(runner):
        for _step in range(1, max_new_tokens):
            runner.reset_step(total_len - 1)
            with torch.no_grad():
                cache_position = None
                if os.environ.get("CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE") == "1":
                    cache_position = torch.tensor([total_len - 1], device=input_ids.device, dtype=torch.long)
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    cache_position=cache_position,
                )

            logits = outputs.logits.float()
            past_key_values = outputs.past_key_values
            if collect_stats:
                step_metadata.append(runner.summarize())

            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_id)
            if _should_stop(next_id, stop_ids_tensor):
                break

            cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
            step_input_ids = next_id
            total_len += 1

    return torch.cat(generated, dim=1), summarize_condition_block_step_metadata(step_metadata)


def _generate_condition_block_cuda_graph(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    dataset=None,
    term1_mass_exp=False,
):
    """Decode with one CUDA-graph replay per generated token.

    Prefill runs unchanged (SDPA + DynamicCache), the prompt KV moves into a
    StaticCache, a few eager decode steps build the per-layer prompt summaries
    and warm the Triton kernels, then the whole decode step (forward + argmax
    + state advance) is captured once and replayed for the remaining tokens.
    Replays execute no Python, so host/launch overhead drops to a single
    graph-launch per token.

    Constraints: requires CONDITION_BLOCK_SKIP_STATS=1, full_attention_layers=0
    and the fused Triton stage2 path. Stop tokens are applied by truncation
    after generation, which yields exactly the greedy early-stop output.
    """
    if os.environ.get("CONDITION_BLOCK_SKIP_STATS") != "1":
        raise ValueError("CONDITION_BLOCK_CUDA_GRAPH=1 requires CONDITION_BLOCK_SKIP_STATS=1.")
    if int(method.full_attention_layers) != 0:
        raise ValueError("CUDA-graph decode supports full_attention_layers=0 only.")
    if int(method.condition_block_size) not in _FUSED_PAGE_SIZES or any(
        os.environ.get(flag) == "1"
        for flag in (
            "CONDITION_BLOCK_DENSE_STAGE2",
            "CONDITION_BLOCK_COMPACT_SDPA_STAGE2",
            "CONDITION_BLOCK_LEGACY_STAGE2",
        )
    ):
        raise ValueError("CUDA-graph decode requires the fused Triton stage2 path.")

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(method.max_new_tokens)
    if max_new_tokens <= 0:
        output_ids = torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        return output_ids, summarize_condition_block_step_metadata([])

    with model_attention_implementation(model, "sdpa"):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
    past_key_values = _static_cache_from_prefill(
        model.config,
        outputs.past_key_values,
        max_cache_len=prompt_len + max_new_tokens,
    )

    dev = input_ids.device
    out_tokens = torch.zeros(max_new_tokens, dtype=input_ids.dtype, device=dev)
    next_id = torch.argmax(outputs.logits.float()[:, -1, :], dim=-1)
    out_tokens[0] = next_id[0]
    del outputs

    input_ids_buf = next_id.view(1, 1).clone()
    cache_pos_buf = torch.tensor([prompt_len], device=dev, dtype=torch.long)
    step_idx_buf = torch.ones(1, device=dev, dtype=torch.long)

    runner = ConditionBlockDecodeRunner(
        model=model,
        model_config=model.config,
        layer_idx_list=list(range(int(model.config.num_hidden_layers))),
        full_attention_layers=method.full_attention_layers,
        block_size=method.condition_block_size,
        eps=method.condition_eps,
        prompt_len=prompt_len,
        pos=prompt_len,
        prompt_prefix_cache={},
        term1_mass_exp=term1_mass_exp,
    )
    runner.static_suffix = True
    runner.suffix_len_dev.fill_(1)

    def _decode_step():
        step_outputs = model(
            input_ids=input_ids_buf,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            cache_position=cache_pos_buf,
        )
        step_next = torch.argmax(step_outputs.logits.float()[0, -1], dim=-1)
        out_tokens.index_copy_(0, step_idx_buf, step_next.view(1))
        input_ids_buf.copy_(step_next.view(1, 1))
        cache_pos_buf.add_(1)
        step_idx_buf.add_(1)
        runner.suffix_len_dev.add_(1)

    # First eager step builds prompt summaries and JIT-compiles kernels; the
    # side-stream steps stabilize allocations per the torch.cuda.graphs recipe.
    warmup_steps = min(3, max_new_tokens - 1)
    with condition_block_decode_context(runner), _no_causal_mask_context(), torch.no_grad():
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_steps):
                _decode_step()
        torch.cuda.current_stream().wait_stream(side_stream)

        remaining = max_new_tokens - 1 - warmup_steps
        if remaining > 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _decode_step()
            for _ in range(remaining):
                graph.replay()
    torch.cuda.synchronize()

    tokens = out_tokens.tolist()
    stop_token_ids = set(_stop_token_ids(tokenizer, dataset))
    n_keep = len(tokens)
    for idx, token in enumerate(tokens):
        if token in stop_token_ids:
            n_keep = idx + 1
            break
    output_ids = out_tokens[:n_keep].view(1, -1).to(dtype=input_ids.dtype)
    return output_ids, {"condition_block_stats_disabled": True, "cuda_graph_decode": True}


def _static_cache_from_prefill(config, prefill_cache, max_cache_len):
    static_cache = StaticCache(config=config, max_cache_len=int(max_cache_len))
    for layer_idx, source_layer in enumerate(prefill_cache.layers):
        key_states = source_layer.keys
        value_states = source_layer.values
        target_layer = static_cache.layers[layer_idx]
        target_layer.lazy_initialization(key_states[:, :, :1], value_states[:, :, :1])
        seq_len = int(key_states.shape[2])
        target_layer.keys[:, :, :seq_len].copy_(key_states)
        target_layer.values[:, :, :seq_len].copy_(value_states)
    return static_cache


def _generate_condition_block_with_hf_loop(
    *, model, tokenizer, input_ids, attention_mask, method, dataset
):
    prompt_len = int(input_ids.shape[1])
    runner = ConditionBlockDecodeRunner(
        model=model,
        model_config=model.config,
        layer_idx_list=list(range(int(model.config.num_hidden_layers))),
        full_attention_layers=method.full_attention_layers,
        block_size=method.condition_block_size,
        eps=method.condition_eps,
        prompt_len=prompt_len,
        pos=prompt_len - 1,
        prompt_prefix_cache={},
    )
    generate_kwargs = {}
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_ids[-1]]
    with condition_block_decode_context(runner), torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(method.max_new_tokens),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    return output_ids[:, prompt_len:], {"condition_block_stats_disabled": True}


def _stop_token_ids(tokenizer, dataset):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            ids.append(int(newline_ids[-1]))
    return ids


def _should_stop(next_id, stop_ids_tensor):
    if stop_ids_tensor is None:
        return False
    return bool(torch.isin(next_id, stop_ids_tensor).all().item())


def _pad_blocks(x, block_size):
    n_heads, seq_len = x.shape[:2]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *x.shape[2:]),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *x.shape[2:]), n_blocks


def _ensure_paged_layout(x, block_size):
    """Keep the (head, block, token, dim) view kernel-addressable via strides.

    The fused kernel reads token pages as ``head * head_stride + token * head_dim``,
    which only requires the inner three dims to be laid out like a contiguous
    tensor. Cache slices satisfy this as views; anything else is materialized
    once here instead of being cloned by ``.contiguous()`` on every decode step.
    """
    head_dim = int(x.shape[-1])
    if x.stride(3) == 1 and x.stride(2) == head_dim and x.stride(1) == block_size * head_dim:
        return x
    return x.contiguous()


def _summary_storage_dtypes(source_dtype):
    """Choose summary storage dtypes without changing fused-kernel arithmetic.

    The exact mixed layout keeps ``k_bar`` in FP32, while ``v_bar`` is rounded
    to BF16 immediately before the representative-value dot product.  The
    coordinate extrema are values copied from the source K tensor, so storing
    them in that source dtype is lossless.  The optional BF16 ``k_bar`` layout
    reduces IO further, but is approximate because the block mean was
    accumulated in FP32.

    ``CONDITION_BLOCK_SUMMARY_DTYPE`` remains the older uniform, potentially
    approximate layout.  Mixing the two modes would make the accuracy contract
    ambiguous, so reject that combination explicitly.
    """
    summary_dtype_name = os.environ.get("CONDITION_BLOCK_SUMMARY_DTYPE", "float32")
    summary_dtype = getattr(torch, summary_dtype_name)
    mixed = os.environ.get("CONDITION_BLOCK_MIXED_SUMMARIES") == "1"
    k_bar_dtype_name = os.environ.get("CONDITION_BLOCK_K_BAR_DTYPE", "float32")
    if k_bar_dtype_name not in ("float32", "bfloat16"):
        raise ValueError(
            "CONDITION_BLOCK_K_BAR_DTYPE must be float32 or bfloat16, got "
            f"{k_bar_dtype_name!r}."
        )
    if not mixed and k_bar_dtype_name != "float32":
        raise ValueError(
            "CONDITION_BLOCK_K_BAR_DTYPE=bfloat16 requires "
            "CONDITION_BLOCK_MIXED_SUMMARIES=1."
        )
    if not mixed:
        return summary_dtype, summary_dtype, summary_dtype
    if summary_dtype != torch.float32:
        raise ValueError(
            "CONDITION_BLOCK_MIXED_SUMMARIES=1 requires "
            "CONDITION_BLOCK_SUMMARY_DTYPE=float32 (or unset)."
        )
    return getattr(torch, k_bar_dtype_name), torch.bfloat16, source_dtype


def _build_prompt_blocks(k_all, v_all, block_size):
    k_block_attn, n_blocks = _pad_blocks(k_all, block_size)
    v_block_attn, _ = _pad_blocks(v_all, block_size)
    k_block_attn = _ensure_paged_layout(k_block_attn, block_size)
    v_block_attn = _ensure_paged_layout(v_block_attn, block_size)
    device = k_all.device
    seq_len = int(k_all.shape[1])

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(n_blocks, block_size)
    valid_token = token_idx < seq_len
    valid = valid_token.view(1, n_blocks, block_size, 1)
    size = valid_token.sum(dim=1).long()
    size_float = size.clamp_min(1).float()

    # Accumulate summaries in FP32 without materializing full FP32 copies of
    # prompt K/V. Token pages stay in their model dtype for the Triton kernel.
    k_sum = (k_block_attn * valid).sum(dim=2, dtype=torch.float32)
    v_sum = (v_block_attn * valid).sum(dim=2, dtype=torch.float32)
    k_for_max = k_block_attn.masked_fill(~valid, float("-inf"))
    k_for_min = k_block_attn.masked_fill(~valid, float("inf"))
    v_norm = torch.linalg.vector_norm(v_block_attn, dim=-1, dtype=torch.float32)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    k_bar_dtype, v_bar_dtype, bound_dtype = _summary_storage_dtypes(
        k_block_attn.dtype
    )
    k_max = k_for_max.amax(dim=2).to(bound_dtype)
    k_min = k_for_min.amin(dim=2).to(bound_dtype)
    prefix = {
        "k_block_attn": k_block_attn,
        "v_block_attn": v_block_attn,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_bar": (k_sum / size_float.view(1, n_blocks, 1)).to(k_bar_dtype),
        "v_bar": (v_sum / size_float.view(1, n_blocks, 1)).to(v_bar_dtype),
        "k_max": k_max,
        "k_min": k_min,
        "v_norm_max": v_norm.amax(dim=2),
        "v_norm_all": v_norm.amax(dim=2).amax(dim=-1),
        "block_valid_counts": size,
    }
    if os.environ.get("CONDITION_BLOCK_TMA_BOUNDS") == "1":
        k_bounds = (
            torch.stack((k_max, k_min), dim=-1).flatten(2).contiguous()
        )
        # Keep compatibility views without retaining duplicate max/min storage.
        # The TMA path consumes k_bounds; the views are only for diagnostics.
        bounds_view = k_bounds.view(*k_max.shape, 2)
        prefix["k_bounds"] = k_bounds
        prefix["k_max"] = bounds_view[..., 0]
        prefix["k_min"] = bounds_view[..., 1]
    return prefix


def _select_prompt_blocks(q_grouped, prefix, eps):
    if q_grouped.is_cuda and os.environ.get("CONDITION_BLOCK_EAGER_SELECTION") != "1":
        return _select_prompt_blocks_triton(q_grouped, prefix, eps)
    return _select_prompt_blocks_eager(q_grouped, prefix, eps)


def _select_prompt_blocks_eager(q_grouped, prefix, eps):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    size = prefix["block_valid_counts"].view(1, 1, -1).long()
    cluster_exists = size > 0
    cluster_exists_view = cluster_exists.view(1, 1, 1, -1)
    size_float = size.clamp_min(1).float()

    k_bar = prefix["k_bar"].to(q_grouped.dtype)
    k_max = prefix["k_max"].to(q_grouped.dtype)
    k_min = prefix["k_min"].to(q_grouped.dtype)
    s_c = torch.einsum("gsqd,gbd->gsqb", q_grouped, k_bar) / scale
    q_bounds = q_grouped[:, :, :, None, :]
    upper = torch.maximum(
        q_bounds * k_max[:, None, None],
        q_bounds * k_min[:, None, None],
    ).sum(dim=-1) / scale
    lower = torch.minimum(
        q_bounds * k_max[:, None, None],
        q_bounds * k_min[:, None, None],
    ).sum(dim=-1) / scale
    delta = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
    delta = delta.masked_fill(~cluster_exists_view, 0.0)

    b_c = prefix["v_norm_max"][:, None, None, :].expand(n_kv_heads, group_size, n_query, -1)
    b_c = b_c.masked_fill(~cluster_exists_view, 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).unsqueeze(2) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists_view, float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )

    selected = (condition.mean(dim=1, keepdim=True) > eps) & cluster_exists_view
    selected = selected.expand(n_kv_heads, group_size, n_query, -1)
    v_bar = prefix["v_bar"][:, None, None].expand(
        n_kv_heads,
        group_size,
        n_query,
        -1,
        head_dim,
    )
    return selected, z_logits, v_bar, size.view(1, -1), cluster_exists.view(1, -1)


def _workspace_empty(workspace, key, shape, *, device, dtype):
    if workspace is None:
        return torch.empty(shape, device=device, dtype=dtype)
    tensor = workspace.get(key)
    if (
        tensor is None
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        workspace[key] = tensor
    return tensor


def _run_condition_block_selection_stats(
    q_grouped,
    prefix,
    workspace=None,
    reduce_globals=True,
    term1_mass_exp=False,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        raise ValueError("Triton selection stats expects q_len=1.")
    n_blocks = int(prefix["block_valid_counts"].numel())
    rows = int(n_kv_heads * group_size)
    q = q_grouped.reshape(rows, head_dim).contiguous()
    selection_chunk = _SELECT_CHUNK
    n_chunks = triton.cdiv(n_blocks, selection_chunk)
    s_cache = _workspace_empty(
        workspace,
        "selection_s",
        (rows, n_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    delta_cache = _workspace_empty(
        workspace,
        "selection_delta",
        (rows, n_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    partial = _workspace_empty(
        workspace,
        "selection_partial",
        (4, rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )
    use_tma_bounds = os.environ.get("CONDITION_BLOCK_TMA_BOUNDS") == "1"
    if use_tma_bounds:
        capability = torch.cuda.get_device_capability(q.device)
        if capability[0] < 9:
            raise ValueError("TMA bounds require Hopper/Blackwell (compute capability >= 9.0).")
        block_d = triton.next_power_of_2(head_dim)
        if block_d != head_dim:
            raise ValueError("TMA bounds currently require a power-of-two head_dim.")
        persist_chunks = int(os.environ.get("CONDITION_BLOCK_TMA_PERSIST_CHUNKS", "1"))
        if persist_chunks < 1:
            raise ValueError("CONDITION_BLOCK_TMA_PERSIST_CHUNKS must be >= 1.")
        bounds = prefix.get("k_bounds")
        if bounds is None:
            raise ValueError("TMA bounds require a prompt prefix with packed k_bounds.")
        bounds_desc = TensorDescriptor.from_tensor(
            bounds,
            block_shape=[1, selection_chunk, 2 * block_d],
        )
        condition_block_selection_stats_tma_kernel[
            (n_kv_heads, triton.cdiv(n_chunks, persist_chunks))
        ](
            q,
            prefix["k_bar"].contiguous(),
            bounds_desc,
            prefix["block_valid_counts"].contiguous(),
            s_cache,
            delta_cache,
            partial[0],
            partial[1],
            partial[2],
            partial[3],
            n_blocks,
            n_chunks,
            group_size,
            head_dim,
            head_dim**-0.5,
            BLOCK_G=triton.next_power_of_2(group_size),
            BLOCK_B=selection_chunk,
            BLOCK_D=block_d,
            PERSIST_CHUNKS=persist_chunks,
            TERM1_MASS_EXP=bool(term1_mass_exp),
            num_warps=_select_warps(n_blocks),
        )
    else:
        _condition_block_selection_stats_kernel[(n_kv_heads, n_chunks)](
            q,
            prefix["k_bar"].contiguous(),
            prefix["k_max"].contiguous(),
            prefix["k_min"].contiguous(),
            prefix["block_valid_counts"].contiguous(),
            s_cache,
            delta_cache,
            partial[0],
            partial[1],
            partial[2],
            partial[3],
            n_blocks,
            n_chunks,
            group_size,
            head_dim,
            head_dim**-0.5,
            BLOCK_G=triton.next_power_of_2(group_size),
            BLOCK_B=selection_chunk,
            BLOCK_D=triton.next_power_of_2(head_dim),
            TERM1_MASS_EXP=bool(term1_mass_exp),
            num_warps=_select_warps(n_blocks),
        )
    global_stats = None
    if reduce_globals:
        global_stats = _workspace_empty(
            workspace,
            "selection_global",
            (4, rows),
            device=q.device,
            dtype=torch.float32,
        )
        _condition_block_selection_reduce_kernel[(rows,)](
            partial[0],
            partial[1],
            partial[2],
            partial[3],
            global_stats[0],
            global_stats[1],
            global_stats[2],
            global_stats[3],
            n_chunks,
            BLOCK_C=triton.next_power_of_2(n_chunks),
            num_warps=4,
        )
    return q, s_cache, delta_cache, partial, global_stats, n_blocks, n_chunks


def _select_prompt_blocks_triton(q_grouped, prefix, eps, term1_mass_exp=False):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        return _select_prompt_blocks_eager(q_grouped, prefix, eps)
    q, s_cache, delta_cache, _partial, global_stats, n_blocks, n_chunks = (
        _run_condition_block_selection_stats(
            q_grouped,
            prefix,
            term1_mass_exp=term1_mass_exp,
        )
    )
    rows = int(n_kv_heads * group_size)
    selection_chunk = _SELECT_CHUNK
    selected = torch.empty((rows, n_blocks), device=q.device, dtype=torch.bool)
    z_logits = torch.empty((rows, n_blocks), device=q.device, dtype=torch.float32)
    _condition_block_selection_finalize_kernel[(n_kv_heads, n_chunks)](
        s_cache,
        delta_cache,
        prefix["v_norm_max"].contiguous(),
        prefix["v_norm_all"].contiguous(),
        prefix["block_valid_counts"].contiguous(),
        global_stats[0],
        global_stats[1],
        global_stats[2],
        global_stats[3],
        selected,
        z_logits,
        n_blocks,
        n_chunks,
        group_size,
        float(eps),
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_B=selection_chunk,
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=4,
    )
    selected = selected.reshape(n_kv_heads, group_size, 1, n_blocks)
    z_logits = z_logits.reshape(n_kv_heads, group_size, 1, n_blocks)
    size = prefix["block_valid_counts"].view(1, -1)
    cluster_exists = size > 0
    v_bar = prefix["v_bar"][:, None, None].expand(
        n_kv_heads, group_size, 1, n_blocks, head_dim
    )
    return selected, z_logits, v_bar, size, cluster_exists


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _condition_block_selection_stats_kernel(
    q_ptr,
    k_bar_ptr,
    k_max_ptr,
    k_min_ptr,
    counts_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    n_blocks,
    n_chunks,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    b_mask = b < n_blocks
    d_mask = d < head_dim
    row = kv_head * group_size + g
    q = tl.load(q_ptr + row[:, None] * head_dim + d[None, :], mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
    k_bar = tl.load(k_bar_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    k_hi = tl.load(k_max_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    k_lo = tl.load(k_min_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
    upper_k = tl.where(q[:, None, :] >= 0.0, k_hi[None, :, :], k_lo[None, :, :])
    lower_k = tl.where(q[:, None, :] >= 0.0, k_lo[None, :, :], k_hi[None, :, :])
    upper = tl.sum(q[:, None, :] * upper_k, axis=2) * scale
    lower = tl.sum(q[:, None, :] * lower_k, axis=2) * scale
    delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    active_2d = g_mask[:, None] & active[None, :]
    z = tl.where(active_2d, z, -float("inf"))
    if TERM1_MASS_EXP:
        # term1 = 2 B softmax(z + delta): no cosh or ``-1`` path.
        zc = z + delta
    else:
        cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
        zc = z + tl.log(cosh_delta)
    tl.store(s_cache_ptr + row[:, None] * n_blocks + b[None, :], s, mask=active_2d)
    tl.store(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], delta, mask=active_2d)
    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=g_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=g_mask)
    tl.store(c_m_ptr + partial_off, c_m, mask=g_mask)
    tl.store(c_l_ptr + partial_off, c_l, mask=g_mask)


@triton.jit(
    do_not_specialize=["n_chunks"],
    do_not_specialize_on_alignment=["n_chunks"],
)
def _condition_block_selection_reduce_kernel(
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    global_z_m_ptr,
    global_z_l_ptr,
    global_c_m_ptr,
    global_c_l_ptr,
    n_chunks,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    mask = c < n_chunks
    z_m = tl.load(z_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    z_l = tl.load(z_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    c_l = tl.load(c_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    global_z_m = tl.max(z_m, axis=0)
    global_c_m = tl.max(c_m, axis=0)
    global_z_l = tl.sum(z_l * tl.exp(z_m - global_z_m), axis=0)
    global_c_l = tl.sum(c_l * tl.exp(c_m - global_c_m), axis=0)
    tl.store(global_z_m_ptr + row, global_z_m)
    tl.store(global_z_l_ptr + row, global_z_l)
    tl.store(global_c_m_ptr + row, global_c_m)
    tl.store(global_c_l_ptr + row, global_c_l)


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _condition_block_selection_finalize_kernel(
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    selected_ptr,
    z_out_ptr,
    n_blocks,
    n_chunks,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    g_mask = g < group_size
    b_mask = b < n_blocks
    row = kv_head * group_size + g
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    mask_2d = g_mask[:, None] & active[None, :]
    s = tl.load(s_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0)
    delta = tl.load(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0)
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    z_m = tl.load(z_m_ptr + row, mask=g_mask, other=0.0)
    z_l = tl.load(z_l_ptr + row, mask=g_mask, other=1.0)
    c_m = tl.load(c_m_ptr + row, mask=g_mask, other=0.0)
    c_l = tl.load(c_l_ptr + row, mask=g_mask, other=1.0)
    b_c = tl.load(v_norm_ptr + kv_head * n_blocks + b, mask=active, other=0.0)
    b_all = tl.load(v_norm_all_ptr + kv_head)
    if TERM1_MASS_EXP:
        exp_neg_delta = tl.exp(-delta)
        tanh_half = (1.0 - exp_neg_delta) / (1.0 + exp_neg_delta)
        term1 = 2.0 * b_all * tl.exp(z + delta - c_m[:, None]) / c_l[:, None]
    else:
        cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
        tanh_half = 2.0 / (1.0 + tl.exp(-delta)) - 1.0
        term1 = 2.0 * b_all * tl.exp(z - c_m[:, None]) * (cosh_delta - 1.0) / c_l[:, None]
    term2 = 2.0 * b_c[None, :] * tl.exp(z - z_m[:, None]) * tanh_half / z_l[:, None]
    condition = tl.where(mask_2d, term1 + term2, 0.0)
    selected = (tl.sum(condition, axis=0) / group_size) > eps
    tl.store(selected_ptr + row[:, None] * n_blocks + b[None, :], selected[None, :], mask=g_mask[:, None] & b_mask[None, :])
    tl.store(z_out_ptr + row[:, None] * n_blocks + b[None, :], z, mask=g_mask[:, None] & b_mask[None, :])


def _condition_block_decode_output(
    *,
    q_grouped,
    prompt_prefix,
    k_suffix,
    v_suffix,
    suffix_len_dev,
    block_size,
    eps,
    prompt_len,
    attention_dtype,
    workspace=None,
    term1_mass_exp=False,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    collect_stats = os.environ.get("CONDITION_BLOCK_SKIP_STATS") != "1"
    use_fused_triton = (
        q_grouped.is_cuda
        and n_query == 1
        and int(block_size) in _FUSED_PAGE_SIZES
        and os.environ.get("CONDITION_BLOCK_DENSE_STAGE2") != "1"
        and os.environ.get("CONDITION_BLOCK_COMPACT_SDPA_STAGE2") != "1"
        and os.environ.get("CONDITION_BLOCK_LEGACY_STAGE2") != "1"
    )
    if os.environ.get("CONDITION_BLOCK_MIXED_SUMMARIES") == "1" and not use_fused_triton:
        raise ValueError(
            "CONDITION_BLOCK_MIXED_SUMMARIES=1 requires the fused Triton "
            "stage2 path with block_size 16, 32, or 64."
        )
    if os.environ.get("CONDITION_BLOCK_TMA_BOUNDS") == "1" and not use_fused_triton:
        raise ValueError(
            "CONDITION_BLOCK_TMA_BOUNDS=1 requires the fused Triton stage2 "
            "path with block_size 16, 32, or 64."
        )
    if term1_mass_exp and not use_fused_triton:
        raise ValueError(
            "term1-softmax requires the fused Triton stage2 path with "
            "block_size 16, 32, or 64."
        )
    if use_fused_triton:
        output, selected = _condition_block_decode_output_fused_triton(
            q_grouped=q_grouped,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            suffix_len_dev=suffix_len_dev,
            eps=eps,
            page_size=int(block_size),
            store_selected=collect_stats,
            output_dtype=attention_dtype,
            workspace=workspace,
            term1_mass_exp=term1_mass_exp,
        )
        size = prompt_prefix["block_valid_counts"].view(1, -1)
        cluster_exists = size > 0
    else:
        # The legacy PyTorch paths historically operate on FP32 Q.  The fused
        # Triton path loads BF16 Q directly and converts in registers, avoiding
        # a standalone cast kernel on every layer and decode step.
        q_grouped = q_grouped.float()
        selected, z_logits, v_bar, size, cluster_exists = _select_prompt_blocks(
            q_grouped,
            prompt_prefix,
            eps,
        )

    if use_fused_triton:
        pass
    elif os.environ.get("CONDITION_BLOCK_DENSE_STAGE2") == "1":
        pos_tensor = torch.tensor(
            [int(prompt_len) + int(k_suffix.shape[1]) - 1],
            device=q_grouped.device,
            dtype=torch.long,
        )
        output = _condition_block_decode_output_dense(
            q_grouped=q_grouped,
            pos_tensor=pos_tensor,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix.float(),
            v_suffix=v_suffix.float(),
            block_size=block_size,
            prompt_len=prompt_len,
            selected=selected,
            z_logits=z_logits,
            v_bar=v_bar,
            cluster_exists=cluster_exists,
        )
    elif os.environ.get("CONDITION_BLOCK_COMPACT_SDPA_STAGE2") == "1":
        pos_tensor = torch.tensor(
            [int(prompt_len) + int(k_suffix.shape[1]) - 1],
            device=q_grouped.device,
            dtype=torch.long,
        )
        output = _condition_block_decode_output_compact_sdpa(
            q_grouped=q_grouped,
            pos_tensor=pos_tensor,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix.float(),
            v_suffix=v_suffix.float(),
            block_size=block_size,
            prompt_len=prompt_len,
            selected=selected,
            attention_dtype=attention_dtype,
        )
    else:
        output = _condition_block_decode_output_triton(
            q_grouped=q_grouped,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            selected=selected,
            z_logits=z_logits,
            attention_dtype=attention_dtype,
        )

    stats = None
    if collect_stats:
        stats = _condition_stats(
            selected=selected,
            size=size,
            cluster_exists=cluster_exists,
            prompt_len=prompt_len,
            suffix_len=int(k_suffix.shape[1]),
        )
    return output, stats


def _condition_block_decode_output_fused_triton(
    *,
    q_grouped,
    prompt_prefix,
    k_suffix,
    v_suffix,
    suffix_len_dev,
    eps,
    page_size,
    store_selected,
    output_dtype,
    workspace=None,
    term1_mass_exp=False,
):
    n_kv_heads, group_size, _n_query, head_dim = q_grouped.shape
    q, s_cache, delta_cache, sel_partial, _global_stats, n_blocks, n_sel_chunks = (
        _run_condition_block_selection_stats(
            q_grouped,
            prompt_prefix,
            workspace,
            reduce_globals=False,
            term1_mass_exp=term1_mass_exp,
        )
    )
    rows = int(n_kv_heads * group_size)
    n_chunks = triton.cdiv(n_blocks, _FINALIZE_CHUNK)
    partial_acc = _workspace_empty(
        workspace,
        "attention_partial_acc",
        (rows, n_chunks, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    partial_m = _workspace_empty(
        workspace,
        "attention_partial_m",
        (rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )
    partial_l = _workspace_empty(
        workspace,
        "attention_partial_l",
        (rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )
    selected_rows = _workspace_empty(
        workspace,
        "selected_rows" if store_selected else "selected_dummy",
        (rows, n_blocks) if store_selected else (1,),
        device=q.device,
        dtype=torch.bool,
    )
    k_block = prompt_prefix["k_block_attn"]
    v_block = prompt_prefix["v_block_attn"]
    _condition_block_finalize_attention_kernel[(n_kv_heads, n_chunks)](
        q,
        k_block,
        v_block,
        prompt_prefix["v_bar"].contiguous(),
        s_cache,
        delta_cache,
        prompt_prefix["v_norm_max"].contiguous(),
        prompt_prefix["v_norm_all"].contiguous(),
        prompt_prefix["block_valid_counts"].contiguous(),
        sel_partial[0],
        sel_partial[1],
        sel_partial[2],
        sel_partial[3],
        k_suffix,
        v_suffix,
        selected_rows,
        partial_acc,
        partial_m,
        partial_l,
        n_blocks,
        n_sel_chunks,
        suffix_len_dev,
        k_block.stride(0),
        v_block.stride(0),
        k_suffix.stride(0),
        k_suffix.stride(1),
        v_suffix.stride(0),
        v_suffix.stride(1),
        group_size,
        head_dim,
        n_chunks,
        float(eps),
        head_dim**-0.5,
        BLOCK_M=16,
        BLOCK_N=_FINALIZE_CHUNK,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_SC=triton.next_power_of_2(n_sel_chunks),
        PAGE_SIZE=int(page_size),
        STORE_SELECTED=bool(store_selected),
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=_FINALIZE_WARPS,
    )
    out = _workspace_empty(
        workspace,
        "attention_output",
        (rows, head_dim),
        device=q.device,
        dtype=output_dtype,
    )
    _condition_block_stage2_reduce_kernel[(rows,)](
        partial_acc,
        partial_m,
        partial_l,
        out,
        n_chunks,
        head_dim,
        BLOCK_C=triton.next_power_of_2(n_chunks),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    selected = None
    if store_selected:
        selected = selected_rows.reshape(n_kv_heads, group_size, 1, n_blocks)
    return out.reshape(n_kv_heads, group_size, 1, head_dim), selected


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
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=float(dropout) if module.training else 0.0,
        scaling=scaling,
        **_kwargs,
    )


def _sdpa_prefill_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **_kwargs,
):
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(dropout) if module.training else 0.0,
        is_causal=True,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return attn_output.transpose(1, 2).contiguous(), None


class ConditionBlockDecodeRunner:
    def __init__(
        self,
        *,
        model,
        model_config,
        layer_idx_list,
        full_attention_layers,
        block_size,
        eps,
        prompt_len,
        pos,
        prompt_prefix_cache,
        term1_mass_exp=False,
    ):
        self.model_config = model_config
        self.module_to_layer_idx = {
            id(layer.self_attn): int(layer_idx)
            for layer_idx, layer in enumerate(model.model.layers)
        }
        self.layer_idx_set = {int(layer_idx) for layer_idx in layer_idx_list}
        self.full_attention_layers = int(full_attention_layers)
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.prompt_len = int(prompt_len)
        self.pos = int(pos)
        self.prompt_prefix_cache = prompt_prefix_cache
        self.term1_mass_exp = bool(term1_mass_exp)
        self.workspace_by_layer = {}
        self.stats_by_layer = {}
        self.aggregate_stats = {}
        self.skip_stats = os.environ.get("CONDITION_BLOCK_SKIP_STATS") == "1"
        # Device-side suffix length consumed by the fused kernel. The eager
        # loop refreshes it in reset_step(); the CUDA-graph loop advances it
        # in-graph so replays never touch Python.
        self.suffix_len_dev = torch.zeros(
            (), dtype=torch.int32, device=next(model.parameters()).device
        )
        # When True, the hook hands the kernel full-size static-cache views and
        # trusts suffix_len_dev for validity, keeping all shapes step-invariant
        # (required for CUDA-graph capture).
        self.static_suffix = False

    def should_compress(self, layer_idx):
        return int(layer_idx) in self.layer_idx_set and int(layer_idx) >= self.full_attention_layers

    def summarize(self):
        return _summarize_stats_lazy(self.aggregate_stats, self.stats_by_layer)

    def reset_step(self, pos):
        self.pos = int(pos)
        self.suffix_len_dev.fill_(max(self.pos + 1 - self.prompt_len, 0))
        self.stats_by_layer.clear()
        self.aggregate_stats.clear()

    def hybrid_attention_forward(
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

        # The persistent HF generate loop enters here for both prompt prefill
        # and decode. Only single-token decode is sparse; position follows the
        # current KV-cache length and therefore needs no Python step runner.
        if query.shape[2] != 1:
            return _sdpa_prefill_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
        if layer_idx is None or not self.should_compress(layer_idx):
            if layer_idx in self.layer_idx_set and not self.skip_stats:
                stats = full_attention_stats_for_heads(query.shape[1], [self.pos])
                self.stats_by_layer[int(layer_idx)] = stats
                _merge_stats_lazy(self.aggregate_stats, stats)
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
            raise ValueError("condition_block fast path expects batch_size=1 and q_len=1.")
        if self.pos < self.prompt_len:
            raise ValueError("condition_block fast path expects generated-token decode positions.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("condition_block fast path expects grouped query attention.")

        q_grouped = query[0].reshape(
            n_kv_heads, n_heads // n_kv_heads, q_len, head_dim
        )
        if self.static_suffix:
            # Full static-cache views: shapes never change across steps; the
            # kernel reads only suffix_len_dev valid suffix tokens.
            k_all = key[0]
            v_all = value[0]
        else:
            visible_len = self.pos + 1
            k_all = key[0, :, :visible_len]
            v_all = value[0, :, :visible_len]
        cache_key = (int(layer_idx), self.prompt_len, self.block_size)
        prompt_prefix = self.prompt_prefix_cache.get(cache_key)
        if prompt_prefix is None:
            prompt_prefix = _build_prompt_blocks(
                k_all[:, : self.prompt_len],
                v_all[:, : self.prompt_len],
                self.block_size,
            )
            self.prompt_prefix_cache[cache_key] = prompt_prefix

        output, stats = _condition_block_decode_output(
            q_grouped=q_grouped,
            prompt_prefix=prompt_prefix,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            suffix_len_dev=self.suffix_len_dev,
            block_size=self.block_size,
            eps=self.eps,
            prompt_len=self.prompt_len,
            attention_dtype=query.dtype,
            workspace=self.workspace_by_layer.setdefault(int(layer_idx), {}),
            term1_mass_exp=self.term1_mass_exp,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)

        if stats is not None:
            self.stats_by_layer[int(layer_idx)] = stats
            _merge_stats_lazy(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def _no_causal_mask_context():
    """Skip HF causal-mask construction for the graphed decode step.

    All layers run the sparse path, which ignores the mask, and HF's
    ``eager_mask`` builds it with ``torch.tensor(scalar, device=...)`` — a
    host-to-device copy that is illegal while a CUDA graph is capturing.
    """
    original = modeling_llama.create_causal_mask
    modeling_llama.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield
    finally:
        modeling_llama.create_causal_mask = original


@contextlib.contextmanager
def condition_block_decode_context(runner):
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = runner.hybrid_attention_forward
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


@contextlib.contextmanager
def model_attention_implementation(model, implementation):
    original = model.config._attn_implementation
    model.config._attn_implementation = implementation
    try:
        yield
    finally:
        model.config._attn_implementation = original


def _merge_stats_lazy(total, add):
    for key, value in add.items():
        if key in total:
            total[key] = total[key] + value
        else:
            total[key] = value


def _to_python_number(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor in condition-block stats.")
        return int(value.item())
    return int(value)


def _stats_ratio(numerator, denominator):
    num = _to_python_number(numerator)
    den = max(_to_python_number(denominator), 1)
    return float(num / den)


def _materialize_stats(stats):
    return {key: _to_python_number(value) for key, value in stats.items()}


def _summarize_stats_lazy(aggregate_stats, stats_by_layer):
    # This function is called once per decode step. Keep it synchronization-free:
    # raw scalar tensors are materialized only after generation finishes in
    # summarize_condition_block_step_metadata().
    return {
        "aggregate": dict(aggregate_stats),
        "by_layer": {
            int(layer_idx): dict(stats) for layer_idx, stats in stats_by_layer.items()
        },
    }


def _condition_stats(*, selected, size, cluster_exists, prompt_len, suffix_len):
    n_kv_heads, group_size, n_query = selected.shape[:3]
    size_view = size.view(1, 1, n_query, -1)
    selected_tokens = (selected.long() * size_view).sum()
    cluster_active = (~selected) & cluster_exists.view(1, 1, n_query, -1)
    suffix_tokens = int(max(int(suffix_len), 0) * n_kv_heads * group_size)
    n_rows = int(n_kv_heads * group_size * n_query)
    total_available = int((int(prompt_len) + max(int(suffix_len), 0)) * n_rows)
    return {
        "rows": n_rows,
        "clusters": cluster_exists.sum() * n_kv_heads * group_size,
        "selected_clusters": selected.sum(),
        "selected_tokens": selected_tokens,
        "hybrid_tokens": selected_tokens + cluster_active.sum() + suffix_tokens,
        "total_available": total_available,
    }


def full_attention_stats(ctx, pos_list):
    return full_attention_stats_for_heads(int(ctx.model_config.num_attention_heads), pos_list)


def full_attention_stats_for_heads(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * int(n_heads)
    return {
        "rows": int(n_heads) * len(pos_list),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


def _full_generation_step_metadata(model, pos_list):
    aggregate = {}
    by_layer = {}
    n_heads = int(model.config.num_attention_heads)
    for layer_idx in range(int(model.config.num_hidden_layers)):
        stats = full_attention_stats_for_heads(n_heads, pos_list)
        by_layer[layer_idx] = stats
        merge_stats(aggregate, stats)
    return summarize_stats(aggregate, by_layer)


def summarize_condition_block_step_metadata(step_metadata):
    if os.environ.get("CONDITION_BLOCK_SKIP_STATS") == "1":
        return {"condition_block_stats_disabled": True}

    aggregate = {}
    by_step = []
    for step_idx, metadata in enumerate(step_metadata):
        if not metadata:
            continue
        step_aggregate = metadata.get("aggregate", {})
        step_hybrid = step_aggregate.get("hybrid_tokens", 0)
        step_total = step_aggregate.get("total_available", 0)
        by_step.append(
            {
                "step": step_idx,
                "equiv_budget": _stats_ratio(step_hybrid, step_total),
            }
        )
        for key, value in step_aggregate.items():
            if key.startswith("mean_"):
                continue
            _merge_stats_lazy(aggregate, {key: value})

    materialized = _materialize_stats(aggregate)
    total_available = max(int(materialized.get("total_available", 0)), 1)
    rows = max(int(materialized.get("rows", 0)), 1)
    hybrid_tokens = int(materialized.get("hybrid_tokens", 0))
    equiv_budget = float(hybrid_tokens / total_available)
    return {
        "condition_block_equiv_budget": equiv_budget,
        "condition_block_budget": {
            **materialized,
            "mean_hybrid_tokens": float(hybrid_tokens / rows),
            "mean_budget_causal": equiv_budget,
            "mean_budget_visible": equiv_budget,
            "by_step": by_step,
        },
    }
