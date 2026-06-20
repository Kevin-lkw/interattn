import math

import torch

from ...context import RunContext
from .attention_topk import run_prefill_only_attention_topk
from .condition_block import build_condition_args, run_prefill_only_condition_block
from .hf import generate_hf
from .kvpress import build_kvpress_press
from .quest import build_quest_args, run_prefill_only_quest


def generate_with_method(model, tokenizer, input_ids, attention_mask, method, device, dataset=None):
    if method.kind == "full":
        return generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset)
    if method.kind.startswith("kvpress_") or method.kind == "h2o":
        press = build_kvpress_press(method)
        with press(model):
            return generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset)
    return generate_with_full_forward_patches(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        method=method,
        device=device,
        dataset=dataset,
    )


def generate_with_full_forward_patches(
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
    prompt_len = int(input_ids.shape[1])
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            stop_token_ids.append(newline_ids[-1])
    for _step in range(method.max_new_tokens):
        logits = next_logits_with_local_method(
            model=model,
            tokenizer=tokenizer,
            input_ids=cur_ids,
            attention_mask=cur_mask,
            prompt_len=prompt_len,
            method=method,
            device=device,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        stop_ids = torch.tensor(stop_token_ids, device=next_id.device)
        if stop_token_ids and bool(torch.isin(next_id, stop_ids).all().item()):
            break
    if not generated:
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=input_ids.dtype)
    return torch.cat(generated, dim=1)


def next_logits_with_local_method(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    prompt_len,
    method,
    device,
):
    seq_len = input_ids.shape[1]
    prompt_len = int(prompt_len)
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
        return run_prefill_only_attention_topk(
            ctx=ctx,
            budget=method.budget,
            full_attention_layers=method.full_attention_layers,
            seq_len=seq_len,
            prompt_len=prompt_len,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    if method.kind == "condition_block":
        args = build_condition_args(method, prompt_len)
        return run_prefill_only_condition_block(
            ctx=ctx,
            args=args,
            eps=method.condition_eps,
            layer_idx_list=list(range(model.config.num_hidden_layers)),
            prompt_len=prompt_len,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    if method.kind == "quest":
        args, page_budget = build_quest_args(method, prompt_len)
        return run_prefill_only_quest(
            ctx=ctx,
            args=args,
            budget=method.budget,
            page_budget=page_budget,
            layer_idx_list=list(range(model.config.num_hidden_layers)),
            prompt_len=prompt_len,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    raise ValueError(f"Unknown local generation method: {method.kind}")
