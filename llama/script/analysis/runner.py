import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import gen_mask, optimize_alpha_star
from .config import parse_args, set_seed, str_to_torch_dtype
from .context import RunContext
from .sanity import (
    build_modified_attn_hidden,
    compute_final_kl_with_reinjected_alpha,
    move_model_inputs_to_device,
    unpack_result_entry,
)


def load_context(args, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
    )

    kv_path = f"../{args.model_name}_{args.dataset}_st{args.start}.pt"
    print(f"Loading KV cache from: {kv_path}")
    kv = torch.load(kv_path, weights_only=False)

    return RunContext(
        model=model,
        tokenizer=tokenizer,
        rope_qkv=kv["after_rope"],
        inputs=kv["input"],
        outputs=kv["output"],
        attn_output=kv["attention_output"],
        layer_input=kv["layer_input"],
        gt_label=kv["gt_label"],
        model_config=kv["model_config"],
        dtype=dtype,
        device=device,
    )


def validate_args_with_cache(ctx, args):
    cache_seq_len = ctx.rope_qkv[0]["q"].shape[2]
    if args.seq_len > cache_seq_len:
        raise ValueError(
            f"--seq-len ({args.seq_len}) exceeds cached sequence length ({cache_seq_len})."
        )
    if args.tail_len > args.seq_len:
        raise ValueError(
            f"--tail-len ({args.tail_len}) must be <= --seq-len ({args.seq_len})."
        )


def run_layer_budgets(ctx, args, layer_idx, head_idx, pos_list, model_inputs, ref_tail_logits):
    save_path = f"../result/layer{layer_idx}/{args.dataset}/{args.strategy}/result.pt"

    if os.path.exists(save_path):
        result = torch.load(save_path)
        print(f"Loaded existing results for layer {layer_idx}. Existing budgets: {list(result.keys())}")
    else:
        result = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"No existing file found for layer {layer_idx}, starting a new one.")

    for budget in args.budgets:
        if budget in result:
            alpha_exist, wrapped_exist = unpack_result_entry(result[budget])
            if args.sanity_check and "sanity_kl" not in wrapped_exist:
                attn_hidden_patch = build_modified_attn_hidden(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=alpha_exist,
                    device=ctx.device,
                )
                sanity_kl = compute_final_kl_with_reinjected_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    attn_hidden_patch=attn_hidden_patch,
                    model_inputs=model_inputs,
                    ref_tail_logits=ref_tail_logits,
                )
                wrapped_exist["sanity_kl"] = sanity_kl
                result[budget] = wrapped_exist
                print(
                    f"Sanity check (existing) layer {layer_idx}, budget {budget}, final KL: {sanity_kl:.6f}"
                )
            else:
                print(f"Budget {budget} already exists in layer {layer_idx}, skipping.")
            continue

        mask = gen_mask(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
            budget=budget,
            prompt_len=args.prompt_len,
            seq_len=args.seq_len,
        )

        print(f"Optimizing alpha_star for layer {layer_idx}, budget {budget}")
        alpha, p_alpha, p_teacher, loss = optimize_alpha_star(
            ctx=ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            training_steps=args.training_steps,
            lr=args.lr,
            mask=mask,
            device=ctx.device,
        )

        print(f"final loss for layer {layer_idx} with budget {budget}: {loss[-1]}")
        entry = {"opt": (alpha, p_alpha, p_teacher, loss)}

        if args.sanity_check:
            attn_hidden_patch = build_modified_attn_hidden(
                ctx=ctx,
                layer_idx=layer_idx,
                head_idx=head_idx,
                pos_list=pos_list,
                alpha=alpha,
                device=ctx.device,
            )
            sanity_kl = compute_final_kl_with_reinjected_alpha(
                ctx=ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                attn_hidden_patch=attn_hidden_patch,
                model_inputs=model_inputs,
                ref_tail_logits=ref_tail_logits,
            )
            entry["sanity_kl"] = sanity_kl
            print(
                f"Sanity check layer {layer_idx}, budget {budget}, final KL gap: {sanity_kl:.6f}"
            )

        result[budget] = entry

    torch.save(result, save_path)
    print(f"Optimization completed and results saved to {save_path}")


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)
    device = args.device

    ctx = load_context(args, dtype=dtype, device=device)
    print("num_hidden_layers:", ctx.model_config.num_hidden_layers)

    validate_args_with_cache(ctx, args)

    n_heads = ctx.model_config.num_attention_heads
    head_idx = list(range(n_heads))
    pos_list = list(range(args.seq_len - args.tail_len, args.seq_len))

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    ctx.model.eval()

    ref_tail_logits = None
    if args.sanity_check:
        with torch.no_grad():
            ref_tail_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
        print("Reference logits computed for sanity check.")

    for layer_idx in args.layers:
        run_layer_budgets(
            ctx=ctx,
            args=args,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            ref_tail_logits=ref_tail_logits,
        )
