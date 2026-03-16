import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import build_qk_routing_alpha, gen_mask, optimize_alpha_star
from .config import parse_args, set_seed, str_to_torch_dtype
from .context import RunContext
from .sanity import (
    build_modified_attn_hidden,
    compute_final_kl_with_reinjected_alpha,
    has_full_sanity_metrics,
    move_model_inputs_to_device,
    unpack_result_entry,
)


def load_context(args, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
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


def resolve_layers(layer_indices, all_layers, num_hidden_layers):
    if all_layers:
        return list(range(num_hidden_layers))

    if layer_indices is None:
        layer_indices = [5, 10, 15, 20, 25, 30]

    layers = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {num_hidden_layers - 1}]"
            )
        layers.append(layer_idx)

    return layers


def get_result_path(layer_idx, dataset, strategy, loss_type):
    return f"../result/layer{layer_idx}/{dataset}/{strategy}/{loss_type}/result.pt"


def has_full_baseline_metrics(entry):
    required_keys = [
        "baseline_sanity_kl",
        "baseline_teacher_nll",
        "baseline_student_nll",
        "baseline_nll_gap",
    ]
    return all(key in entry for key in required_keys)


def run_layer_budgets(ctx, args, layer_idx, head_idx, pos_list, model_inputs, ref_tail_logits):
    save_path = get_result_path(layer_idx, args.dataset, args.strategy, args.loss_type)

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
            mask_for_baseline = None
            if args.sanity_check and not has_full_sanity_metrics(wrapped_exist):
                attn_hidden_patch = build_modified_attn_hidden(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=alpha_exist,
                    device=ctx.device,
                )
                metrics = compute_final_kl_with_reinjected_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    attn_hidden_patch=attn_hidden_patch,
                    model_inputs=model_inputs,
                    ref_tail_logits=ref_tail_logits,
                )
                wrapped_exist.update(metrics)
                result[budget] = wrapped_exist
                print(
                    f"Sanity check (existing) layer {layer_idx}, budget {budget}, "
                    f"KL: {metrics['sanity_kl']:.6f}, "
                    f"teacher NLL: {metrics['teacher_nll']:.6f}, "
                    f"student NLL: {metrics['student_nll']:.6f}, "
                    f"NLL gap: {metrics['nll_gap']:.6f}"
                )
            elif not args.sanity_check:
                print(f"Budget {budget} already exists in layer {layer_idx}, skipping.")

            if args.sanity_check and args.baseline_check and not has_full_baseline_metrics(wrapped_exist):
                if mask_for_baseline is None:
                    mask_for_baseline = gen_mask(
                        ctx=ctx,
                        layer_idx=layer_idx,
                        pos_list=pos_list,
                        head_idx=head_idx,
                        strategy=args.strategy,
                        budget=budget,
                        prompt_len=args.prompt_len,
                        seq_len=args.seq_len,
                    )
                baseline_alpha = build_qk_routing_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    mask=mask_for_baseline,
                    device=ctx.device,
                )
                baseline_patch = build_modified_attn_hidden(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=baseline_alpha,
                    device=ctx.device,
                )
                baseline_metrics = compute_final_kl_with_reinjected_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    attn_hidden_patch=baseline_patch,
                    model_inputs=model_inputs,
                    ref_tail_logits=ref_tail_logits,
                )
                wrapped_exist.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
                if has_full_sanity_metrics(wrapped_exist):
                    wrapped_exist["delta_sanity_kl"] = (
                        wrapped_exist["baseline_sanity_kl"] - wrapped_exist["sanity_kl"]
                    )
                    wrapped_exist["delta_student_nll"] = (
                        wrapped_exist["baseline_student_nll"] - wrapped_exist["student_nll"]
                    )
                    wrapped_exist["delta_nll_gap"] = (
                        wrapped_exist["baseline_nll_gap"] - wrapped_exist["nll_gap"]
                    )
                result[budget] = wrapped_exist
                print(
                    f"Baseline check (existing) layer {layer_idx}, budget {budget}, "
                    f"KL: {baseline_metrics['sanity_kl']:.6f}, "
                    f"teacher NLL: {baseline_metrics['teacher_nll']:.6f}, "
                    f"student NLL: {baseline_metrics['student_nll']:.6f}, "
                    f"NLL gap: {baseline_metrics['nll_gap']:.6f}"
                )
                if has_full_sanity_metrics(wrapped_exist):
                    print(
                        f"Delta(alpha* vs baseline) layer {layer_idx}, budget {budget}: "
                        f"dKL={wrapped_exist['delta_sanity_kl']:.6f}, "
                        f"dStudentNLL={wrapped_exist['delta_student_nll']:.6f}, "
                        f"dNLLGap={wrapped_exist['delta_nll_gap']:.6f}"
                    )
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
            loss_type=args.loss_type,
            device=ctx.device,
        )

        print(f"final loss for layer {layer_idx} with budget {budget}: {loss[-1]}")
        entry = {"opt": (alpha, p_alpha, p_teacher, loss), "loss_type": args.loss_type}

        if args.sanity_check:
            attn_hidden_patch = build_modified_attn_hidden(
                ctx=ctx,
                layer_idx=layer_idx,
                head_idx=head_idx,
                pos_list=pos_list,
                alpha=alpha,
                device=ctx.device,
            )
            metrics = compute_final_kl_with_reinjected_alpha(
                ctx=ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                attn_hidden_patch=attn_hidden_patch,
                model_inputs=model_inputs,
                ref_tail_logits=ref_tail_logits,
            )
            entry.update(metrics)
            print(
                f"Sanity check layer {layer_idx}, budget {budget}, "
                f"KL: {metrics['sanity_kl']:.6f}, "
                f"teacher NLL: {metrics['teacher_nll']:.6f}, "
                f"student NLL: {metrics['student_nll']:.6f}, "
                f"NLL gap: {metrics['nll_gap']:.6f}"
            )

            if args.baseline_check:
                baseline_alpha = build_qk_routing_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    mask=mask,
                    device=ctx.device,
                )
                baseline_patch = build_modified_attn_hidden(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pos_list=pos_list,
                    alpha=baseline_alpha,
                    device=ctx.device,
                )
                baseline_metrics = compute_final_kl_with_reinjected_alpha(
                    ctx=ctx,
                    layer_idx=layer_idx,
                    pos_list=pos_list,
                    attn_hidden_patch=baseline_patch,
                    model_inputs=model_inputs,
                    ref_tail_logits=ref_tail_logits,
                )
                entry.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
                entry["delta_sanity_kl"] = entry["baseline_sanity_kl"] - entry["sanity_kl"]
                entry["delta_student_nll"] = entry["baseline_student_nll"] - entry["student_nll"]
                entry["delta_nll_gap"] = entry["baseline_nll_gap"] - entry["nll_gap"]
                print(
                    f"Baseline check layer {layer_idx}, budget {budget}, "
                    f"KL: {baseline_metrics['sanity_kl']:.6f}, "
                    f"teacher NLL: {baseline_metrics['teacher_nll']:.6f}, "
                    f"student NLL: {baseline_metrics['student_nll']:.6f}, "
                    f"NLL gap: {baseline_metrics['nll_gap']:.6f}"
                )
                print(
                    f"Delta(alpha* vs baseline) layer {layer_idx}, budget {budget}: "
                    f"dKL={entry['delta_sanity_kl']:.6f}, "
                    f"dStudentNLL={entry['delta_student_nll']:.6f}, "
                    f"dNLLGap={entry['delta_nll_gap']:.6f}"
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

    validate_args_with_cache(ctx, args)

    n_heads = ctx.model_config.num_attention_heads
    head_idx = list(range(n_heads))
    pos_list = list(range(args.seq_len - args.tail_len, args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    ctx.model.eval()

    ref_tail_logits = None
    if args.baseline_check and not args.sanity_check:
        raise ValueError("--baseline-check requires --sanity-check because it compares final logits/NLL.")

    if args.sanity_check:
        with torch.no_grad():
            ref_tail_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
        print("Reference logits computed for sanity check.")

    for layer_idx in layer_idx_list:
        run_layer_budgets(
            ctx=ctx,
            args=args,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            ref_tail_logits=ref_tail_logits,
        )
