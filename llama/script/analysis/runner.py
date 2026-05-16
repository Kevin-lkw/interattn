"""
This is the main runner for the online routing experiment. It includes:
1. Loading the model, tokenizer, and dataset context.
2. Running the online optimization for each specified budget and layer.
3. Saving the results after each budget optimization.
4. Running a one-shot multi-layer baseline comparison at the end.
"""

import copy
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import gen_mask, optimize_alpha_star
from .baseline_eval import run_multilayer_baseline_check
from .context import RunContext
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .runner_utils import (
    build_context_with_new_start,
    build_prompt,
    create_base_runner_parser,
    finalize_runner_args,
    normalize_budget_key,
    resolve_head_indices,
    set_seed,
    str_to_torch_dtype,
)
from .sanity import (
    build_modified_attn_hidden,
    get_tail_labels,
    move_model_inputs_to_device,
    compute_metrics,
)


def parse_args():
    parser = create_base_runner_parser(
        description=(
            "Run online optimal routing and one-shot QK-routing baseline comparison."
        ),
        strategy_choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
        default_strategy="h2o",
        strategy_help="Mask generation strategy.",
        eval_start_help=(
            "Start index for the sample to optimize/evaluate. If not set, use --start. "
            "The optimized alpha/patches are sample-specific, so --eval-start is fitted "
            "and evaluated on that sample."
        ),
    )
    parser.set_defaults(
        training_steps=1000,
        budgets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama-2-7b-hf",
        help="Short model name kept for compatibility with older scripts.",
    )
    return finalize_runner_args(parser.parse_args())


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

    prompt = build_prompt(args.dataset)

    # Build one continuous context window and next-token labels directly from raw text.
    encoded = tokenizer(
        prompt,
        max_length=args.start + args.seq_len + 1,
        truncation=True,
        return_tensors="pt",
    )
    total_len = encoded["input_ids"].shape[1]
    required_len = args.start + args.seq_len + 1
    if total_len < required_len:
        raise ValueError(
            f"Tokenized prompt length ({total_len}) is shorter than required ({required_len})."
        )

    inputs = {
        key: value[:, args.start : args.start + args.seq_len]
        for key, value in encoded.items()
    }
    gt_label = encoded["input_ids"][:, args.start + 1 : args.start + args.seq_len + 1]

    return RunContext(
        model=model,
        tokenizer=tokenizer,
        rope_qkv=None,
        inputs=inputs,
        outputs=None,
        attn_output=None,
        layer_input=None,
        gt_label=gt_label,
        model_config=model.config,
        dtype=dtype,
        device=device,
    )


def validate_args_with_cache(ctx, args):
    input_seq_len = ctx.inputs["input_ids"].shape[1]
    if args.seq_len > input_seq_len:
        raise ValueError(
            f"--seq-len ({args.seq_len}) exceeds prepared sequence length ({input_seq_len})."
        )


def resolve_layers(layer_indices, all_layers, num_hidden_layers):
    if all_layers:
        return list(range(num_hidden_layers))

    if layer_indices is None:
        return list(range(num_hidden_layers))

    layers = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {num_hidden_layers - 1}]"
            )
        layers.append(layer_idx)

    return layers


def get_result_path(layer_idx, dataset, start, adaptive_budget, strategy, loss_type):
    adaptive_str = "adaptive" if adaptive_budget else "fixed"
    return f"../result/{dataset}_{start}/{adaptive_str}/{strategy}/{loss_type}/layer{layer_idx}/result.pt"


def load_or_init_layer_results(layer_idx_list, args):
    layer_results = {}
    for layer_idx in layer_idx_list:
        save_path = get_result_path(layer_idx, args.dataset, args.start, args.adaptive_budget, args.strategy, args.loss_type)
        if os.path.exists(save_path):
            result = torch.load(save_path, map_location="cpu", weights_only=False)
            print(
                f"Loaded existing results for layer {layer_idx}. Existing budgets: {list(result.keys())}"
            )
        else:
            result = {}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"No existing file found for layer {layer_idx}, starting a new one.")
        layer_results[layer_idx] = result
    return layer_results


def load_or_init_final_metrics(args):
    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    save_path = f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/{args.loss_type}/layer_all/budget_to_final_metrics.pt"
    if os.path.exists(save_path):
        metrics = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(metrics, dict):
            print(f"Loaded existing final metrics. Existing budgets: {list(metrics.keys())}")
            return metrics
    return {}


def save_results(layer_idx_list, layer_results, budget_to_final_metrics, args):
    for layer_idx in layer_idx_list:
        save_path = get_result_path(layer_idx, args.dataset, args.start, args.adaptive_budget, args.strategy, args.loss_type)
        torch.save(layer_results[layer_idx], save_path)
        print(f"Optimization completed and results saved to {save_path}")
    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    save_path = f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/{args.loss_type}/layer_all/budget_to_final_metrics.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(budget_to_final_metrics, save_path)

def run_budget_online(
    ctx,
    args,
    budget,
    layer_idx_list,
    head_idx,
    pos_list,
    model_inputs,
    layer_results,
):
    # This dict carries patches that are immediately applied to later layers in this budget.
    layer_to_patch = {}

    for layer_idx in layer_idx_list:
        import time
        t0 = time.time()
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

        layer_result = layer_results[layer_idx]
        budget_key = normalize_budget_key(layer_result, budget)
        patch_hidden = None
        mask = None
        alpha = None
        if budget_key is not None:
            existing_entry = layer_result[budget_key]
            if (
                isinstance(existing_entry, dict)
                and existing_entry.get("optimized_online", False)
                and "patch_hidden" in existing_entry
            ):
                patch_hidden = existing_entry["patch_hidden"].to(ctx.device)
                print(f"Layer {layer_idx}, budget {budget} already exists, reuse patch_hidden.")

        if patch_hidden is None:
            mask = gen_mask(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                head_idx=head_idx,
                strategy=args.strategy,
                budget=budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
            )
            print(f"Optimizing alpha_star for layer {layer_idx}, budget {budget}")
            alpha, _p_alpha, _p_teacher, loss = optimize_alpha_star(
                ctx=layer_ctx,
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
            patch_hidden = build_modified_attn_hidden(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                head_idx=head_idx,
                pos_list=pos_list,
                alpha=alpha,
                device=ctx.device,
            )
            patch_hidden = patch_hidden.detach()
            layer_result[float(budget)] = {
                "patch_hidden": patch_hidden.cpu(),
                "loss": loss,
                "loss_type": args.loss_type,
                "optimized_online": True,
            }

        layer_to_patch[layer_idx] = patch_hidden
        del artifacts, layer_ctx
        if mask is not None:
            del mask
        if alpha is not None:
            del alpha
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()
        t1 = time.time()
        print(f"Layer {layer_idx} done for budget {budget} in {t1 - t0:.2f} seconds.")
        print("estimated time for this budget: ", f"{(t1 - t0) * (len(layer_idx_list)) / 60:.2f} minutes")
    return layer_to_patch


def main():
    set_seed(42)
    args = parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    dtype = str_to_torch_dtype(args.dtype)

    base_ctx = load_context(args, dtype=dtype, device=args.device)
    validate_args_with_cache(base_ctx, args)
    base_ctx.model.eval()

    ctx = base_ctx
    run_args = args
    if args.eval_start != args.start:
        ctx = build_context_with_new_start(
            base_ctx=base_ctx,
            dataset_name=args.dataset,
            start=args.eval_start,
            seq_len=args.seq_len,
        )
        validate_args_with_cache(ctx, args)
        run_args = copy.copy(args)
        run_args.start = args.eval_start
        print(
            "[runner] eval_start differs from start; fitting and evaluating "
            "online optimal routing on the eval sample because patches are sample-specific."
        )

    head_idx = resolve_head_indices(ctx.model_config.num_attention_heads)
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)

    ref_tail_logits = None
    labels = None
    
    with torch.no_grad():
        ref_tail_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    print("Reference logits computed for check routines.")
    labels = get_tail_labels(ctx, pos_list, ctx.device)

    layer_results = load_or_init_layer_results(layer_idx_list, run_args)
    budget_to_final_metrics = load_or_init_final_metrics(run_args)

    for budget in args.budgets:
        if normalize_budget_key(budget_to_final_metrics, budget) is not None:
            print(f"[skip] budget={budget} already exists in final metrics, skip.")
            continue

        print(f"\n[online optimize] budget={budget}")
        final_layer_patch = run_budget_online(
            ctx=ctx,
            args=run_args,
            budget=budget,
            layer_idx_list=layer_idx_list,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_results=layer_results,
        )

        student_tail_logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=final_layer_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        final_metrics = compute_metrics(ref_tail_logits, student_tail_logits, labels)
        budget_to_final_metrics[float(budget)] = final_metrics
        print(
            f"[online sanity] budget={budget}: "
            f"KL={final_metrics['sanity_kl']:.6f}, "
            f"teacher NLL={final_metrics['teacher_nll']:.6f}, "
            f"student NLL={final_metrics['student_nll']:.6f}, "
            f"NLL gap={final_metrics['nll_gap']:.6f}"
        )
        save_results(layer_idx_list, layer_results, budget_to_final_metrics, run_args)
        del final_layer_patch, student_tail_logits
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print("Running one-shot multi-layer baseline comparison...")
    run_multilayer_baseline_check(
        ctx=ctx,
        args=run_args,
        target_layers=layer_idx_list,
        head_idx=head_idx,
        pos_list=pos_list,
        model_inputs=model_inputs,
        ref_tail_logits=ref_tail_logits,
    )


if __name__ == "__main__":
    main()
