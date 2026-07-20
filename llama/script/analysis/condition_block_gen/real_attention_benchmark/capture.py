"""Capture real decode-attention tensors from a LongBench model forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..methods.condition_block_triton_impl.core import (
    ConditionBlockDecodeRunner,
    _select_prompt_blocks_triton,
    _static_cache_from_prefill,
    condition_block_decode_context,
    model_attention_implementation,
)


@dataclass
class LayerCapture:
    layer_idx: int
    prompt_tokens: int
    q_grouped: torch.Tensor
    k_all: torch.Tensor
    v_all: torch.Tensor
    prompt_prefix: dict[str, torch.Tensor]
    k_suffix: torch.Tensor
    v_suffix: torch.Tensor
    selected: torch.Tensor

    @property
    def suffix_tokens(self):
        return int(self.k_suffix.shape[1])


class RealTensorCaptureRunner(ConditionBlockDecodeRunner):
    """Run the production attention path and retain selected real layer inputs."""

    def __init__(self, *args, capture_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_layers = set(int(layer) for layer in capture_layers)
        self.captures: dict[int, LayerCapture] = {}

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
        output = super().hybrid_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )
        layer_idx = getattr(module, "layer_idx", None)
        if (
            layer_idx is None
            or int(layer_idx) not in self.capture_layers
            or int(layer_idx) in self.captures
            or query.shape[2] != 1
        ):
            return output

        visible_len = self.pos + 1
        n_kv_heads = int(key.shape[1])
        n_heads = int(query.shape[1])
        head_dim = int(query.shape[-1])
        q_grouped = query[0].reshape(
            n_kv_heads,
            n_heads // n_kv_heads,
            1,
            head_dim,
        ).detach().clone()
        k_all = key[0, :, :visible_len].detach()
        v_all = value[0, :, :visible_len].detach()
        cache_key = (int(layer_idx), self.prompt_len, self.block_size)
        prefix = self.prompt_prefix_cache[cache_key]
        selected_all, *_ = _select_prompt_blocks_triton(
            q_grouped,
            prefix,
            self.eps,
            term1_mass_exp=self.term1_mass_exp,
        )
        selected = selected_all[:, 0, 0].contiguous()
        expected = selected[:, None, None].expand_as(selected_all)
        if not torch.equal(selected_all, expected):
            raise AssertionError("GQA query heads did not share one page mask")
        self.captures[int(layer_idx)] = LayerCapture(
            layer_idx=int(layer_idx),
            prompt_tokens=self.prompt_len,
            q_grouped=q_grouped,
            k_all=k_all,
            v_all=v_all,
            prompt_prefix=prefix,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            selected=selected,
        )
        return output


def capture_first_decode_step(
    *,
    model,
    input_ids,
    attention_mask,
    block_size,
    eps,
    capture_layers,
):
    """Capture the first sparse decode step after an unchanged SDPA prefill."""
    prompt_len = int(input_ids.shape[1])
    with model_attention_implementation(model, "sdpa"), torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    next_id = torch.argmax(prefill.logits.float()[:, -1], dim=-1, keepdim=True)
    cache = _static_cache_from_prefill(
        model.config,
        prefill.past_key_values,
        max_cache_len=prompt_len + 2,
    )
    decode_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
    runner = RealTensorCaptureRunner(
        model=model,
        model_config=model.config,
        layer_idx_list=list(range(int(model.config.num_hidden_layers))),
        full_attention_layers=0,
        block_size=int(block_size),
        eps=float(eps),
        prompt_len=prompt_len,
        pos=prompt_len,
        prompt_prefix_cache={},
        capture_layers=capture_layers,
    )
    runner.reset_step(prompt_len)
    with condition_block_decode_context(runner), torch.no_grad():
        decode = model(
            input_ids=next_id,
            attention_mask=decode_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            cache_position=torch.tensor(
                [prompt_len], device=input_ids.device, dtype=torch.long
            ),
        )
    torch.cuda.synchronize(input_ids.device)
    if set(runner.captures) != set(int(layer) for layer in capture_layers):
        missing = sorted(set(capture_layers) - set(runner.captures))
        raise AssertionError(f"Failed to capture layers: {missing}")
    # Tensors retained in the captures own their storage; the forward outputs
    # and cache container can be released before replay benchmarking.
    del prefill, decode, cache
    return runner.captures
