import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ..context import RunContext
from ..multisample.common import run_attention_topk_method
from ..runner_cond_block import _resolve_block_size, run_for_eps
from ..runner_quest import run_for_budget as run_quest_for_budget


@dataclass
class GenerationMethod:
    name: str
    kind: str
    budget: float = 1.0
    max_new_tokens: int = 32
    full_attention_layers: int = 0
    condition_eps: float = 1.0
    condition_block_size: int | None = None
    condition_delta_mode: str = "range_bound"
    quest_page_size: int = 16
    kvpress_window_size: int = 64
    kvpress_kernel_size: int = 5
    kvpress_alpha_safeguard: float = 0.20
    kvpress_sink_tokens: int = 4

    @property
    def compression_ratio(self):
        return max(0.0, min(1.0, 1.0 - float(self.budget)))


def build_method(args):
    return GenerationMethod(
        name=args.method,
        kind=args.method,
        budget=float(args.budget),
        max_new_tokens=int(args.max_new_tokens),
        full_attention_layers=int(args.full_attention_layers),
        condition_eps=float(args.condition_eps),
        condition_block_size=args.condition_block_size,
        condition_delta_mode=args.condition_delta_mode,
        quest_page_size=int(args.quest_page_size),
        kvpress_window_size=int(args.kvpress_window_size),
        kvpress_kernel_size=int(args.kvpress_kernel_size),
        kvpress_alpha_safeguard=float(args.kvpress_alpha_safeguard),
        kvpress_sink_tokens=int(args.kvpress_sink_tokens),
    )


def add_method_args(parser):
    parser.add_argument(
        "--method",
        default="full",
        choices=[
            "full",
            "kvpress_snapkv",
            "kvpress_adakv_snapkv",
            "kvpress_streamllm",
            "attention_topk",
            "h2o",
            "condition_block",
            "quest",
        ],
    )
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--full-attention-layers", type=int, default=0)
    parser.add_argument("--condition-eps", type=float, default=1.0)
    parser.add_argument("--condition-block-size", type=int, default=None)
    parser.add_argument(
        "--condition-delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--kvpress-window-size", type=int, default=64)
    parser.add_argument("--kvpress-kernel-size", type=int, default=5)
    parser.add_argument("--kvpress-alpha-safeguard", type=float, default=0.20)
    parser.add_argument("--kvpress-sink-tokens", type=int, default=4)
    return parser


def build_kvpress_press(method):
    try:
        from kvpress import (
            AdaKVPress,
            ObservedAttentionPress,
            SnapKVPress,
            StreamingLLMPress,
        )
    except ImportError as exc:
        raise ImportError(
            "kvpress is required for KVPress-backed methods. Install it in the "
            "same environment used to run this script."
        ) from exc

    if method.kind == "h2o":
        return ObservedAttentionPress(compression_ratio=method.compression_ratio)
    if method.kind == "kvpress_snapkv":
        return SnapKVPress(
            compression_ratio=method.compression_ratio,
            window_size=method.kvpress_window_size,
            kernel_size=method.kvpress_kernel_size,
        )
    if method.kind == "kvpress_adakv_snapkv":
        return AdaKVPress(
            SnapKVPress(
                compression_ratio=method.compression_ratio,
                window_size=method.kvpress_window_size,
                kernel_size=method.kvpress_kernel_size,
            ),
            alpha_safeguard=method.kvpress_alpha_safeguard,
        )
    if method.kind == "kvpress_streamllm":
        return StreamingLLMPress(
            compression_ratio=method.compression_ratio,
            n_sink=method.kvpress_sink_tokens,
        )
    raise ValueError(f"Not a KVPress method: {method.kind}")


def generate_with_method(model, tokenizer, input_ids, attention_mask, method, device, dataset=None):
    if method.kind == "full":
        return _generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset)
    if method.kind.startswith("kvpress_") or method.kind == "h2o":
        press = build_kvpress_press(method)
        with press(model):
            return _generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset)
    return _generate_with_full_forward_patches(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        method=method,
        device=device,
        dataset=dataset,
    )


def _generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset=None):
    generate_kwargs = {}
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            generate_kwargs.update(
                {
                    "min_length": input_ids.shape[1] + 1,
                    "eos_token_id": [tokenizer.eos_token_id, newline_ids[-1]],
                }
            )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=method.max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    return output_ids[:, input_ids.shape[1] :]


def _generate_with_full_forward_patches(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    device,
    dataset=None,
):
    generated = []
    cur_ids = input_ids
    cur_mask = attention_mask
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            stop_token_ids.append(newline_ids[-1])
    for _step in range(method.max_new_tokens):
        logits = _next_logits_with_local_method(
            model=model,
            tokenizer=tokenizer,
            input_ids=cur_ids,
            attention_mask=cur_mask,
            method=method,
            device=device,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        if stop_token_ids and bool(torch.isin(next_id, torch.tensor(stop_token_ids, device=next_id.device)).all().item()):
            break
    if not generated:
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=input_ids.dtype)
    return torch.cat(generated, dim=1)


def _next_logits_with_local_method(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    device,
):
    seq_len = input_ids.shape[1]
    ctx = RunContext(
        model=model,
        tokenizer=tokenizer,
        rope_qkv=None,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=None,
        attn_output=None,
        layer_input=None,
        gt_label=input_ids,
        model_config=model.config,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    pos_list = [seq_len - 1]
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    if math.isclose(float(method.budget), 1.0):
        with torch.no_grad():
            return model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()

    if method.kind == "attention_topk":
        logits, _measured = run_attention_topk_method(
            ctx=ctx,
            budget=method.budget,
            full_attention_layers=method.full_attention_layers,
            seq_len=seq_len,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        return logits

    if method.kind == "condition_block":
        args = SimpleNamespace(
            seq_len=seq_len,
            budget=method.budget,
            block_size=method.condition_block_size,
            full_attention_layers=method.full_attention_layers,
            delta_mode=method.condition_delta_mode,
        )
        args.block_size = _resolve_block_size(args)
        logits, _patches, _budget = run_for_eps(
            ctx=ctx,
            args=args,
            eps=method.condition_eps,
            layer_idx_list=list(range(model.config.num_hidden_layers)),
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        return logits

    if method.kind == "quest":
        requested_tokens = max(1, int(seq_len * method.budget))
        page_budget = math.ceil(requested_tokens / method.quest_page_size)
        args = SimpleNamespace(seq_len=seq_len, page_size=method.quest_page_size)
        logits, _patches, _budget = run_quest_for_budget(
            ctx=ctx,
            args=args,
            budget=method.budget,
            page_budget=page_budget,
            layer_idx_list=list(range(model.config.num_hidden_layers)),
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        return logits

    raise ValueError(f"Unknown local generation method: {method.kind}")
