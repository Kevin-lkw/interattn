import os
import time

from numpy import save
import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import gen_mask, optimize_alpha_star
from .baseline_eval import run_multilayer_baseline_check
from .config import parse_args, set_seed, str_to_torch_dtype
from .context import RunContext
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .sanity import (
    build_modified_attn_hidden,
    get_tail_labels,
    move_model_inputs_to_device,
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

    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        prompt = "\n".join([text for text in dataset["text"] if text.strip()])
    elif args.dataset == "pg19":
        dataset = load_dataset("emozilla/pg19-test", split="test")
        prompt = "\n".join([text for text in dataset["text"] if text.strip()])
    else:
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. Supported now: wikitext, pg19"
        )

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
    return f"../result/{dataset}/{strategy}/{loss_type}/layer{layer_idx}/result.pt"


def normalize_budget_key(result_dict, target_budget, atol=1e-12):
    for key in result_dict.keys():
        if abs(float(key) - float(target_budget)) <= atol:
            return key
    return None


def load_or_init_layer_results(layer_idx_list, args):
    layer_results = {}
    for layer_idx in layer_idx_list:
        save_path = get_result_path(layer_idx, args.dataset, args.strategy, args.loss_type)
        if os.path.exists(save_path):
            result = torch.load(save_path, weights_only=False)
            print(
                f"Loaded existing results for layer {layer_idx}. Existing budgets: {list(result.keys())}"
            )
        else:
            result = {}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"No existing file found for layer {layer_idx}, starting a new one.")
        layer_results[layer_idx] = result
    return layer_results


def save_results(layer_idx_list, layer_results, budget_to_final_metrics, args):
    for layer_idx in layer_idx_list:
        save_path = get_result_path(layer_idx, args.dataset, args.strategy, args.loss_type)
        torch.save(layer_results[layer_idx], save_path)
        print(f"Optimization completed and results saved to {save_path}")
    save_path = f"../result/{args.dataset}/{args.strategy}/{args.loss_type}/layer_all/budget_to_final_metrics.pt"
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
            )
            print(f"Optimizing alpha_star for layer {layer_idx}, budget {budget}")
            alpha, p_alpha, p_teacher, loss = optimize_alpha_star(
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
            layer_result[float(budget)] = {
                "patch_hidden": patch_hidden.detach().cpu(),
                "loss": loss,
                "loss_type": args.loss_type,
                "optimized_online": True,
            }

        layer_to_patch[layer_idx] = patch_hidden
        t1 = time.time()
        print(f"Layer {layer_idx} done for budget {budget} in {t1 - t0:.2f} seconds.")
        print("estimated time for this budget: ", f"{(t1 - t0) * (len(layer_idx_list)) / 60:.2f} minutes")
    return layer_to_patch

def compute_metrics(ref_tail_logits, student_tail_logits, labels, unbiased=False):
    # [*, vocab]
    p_teacher = F.softmax(ref_tail_logits, dim=-1)
    logp_teacher = F.log_softmax(ref_tail_logits, dim=-1)
    logp_student = F.log_softmax(student_tail_logits, dim=-1)

    # 逐 token KL: shape = labels.shape
    kl_per_token = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)

    # 逐 token NLL
    teacher_nll_per_token = F.cross_entropy(
        ref_tail_logits.reshape(-1, ref_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    student_nll_per_token = F.cross_entropy(
        student_tail_logits.reshape(-1, student_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    nll_gap_per_token = student_nll_per_token - teacher_nll_per_token

    return {
        "sanity_kl": kl_per_token.mean().item(),
        "sanity_kl_std": kl_per_token.std(unbiased=unbiased).item(),
        "teacher_nll": teacher_nll_per_token.mean().item(),
        "teacher_nll_std": teacher_nll_per_token.std(unbiased=unbiased).item(),
        "student_nll": student_nll_per_token.mean().item(),
        "student_nll_std": student_nll_per_token.std(unbiased=unbiased).item(),
        "nll_gap": nll_gap_per_token.mean().item(),
        "nll_gap_std": nll_gap_per_token.std(unbiased=unbiased).item(),
    }

def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)
    device = args.device

    ctx = load_context(args, dtype=dtype, device=device)

    validate_args_with_cache(ctx, args)

    n_heads = ctx.model_config.num_attention_heads
    head_idx = list(range(n_heads))
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    ctx.model.eval()

    ref_tail_logits = None
    labels = None
    if args.sanity_check or args.baseline_check:
        with torch.no_grad():
            ref_tail_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
        print("Reference logits computed for check routines.")
        labels = get_tail_labels(ctx, pos_list, ctx.device)

    layer_results = load_or_init_layer_results(layer_idx_list, args)
    budget_to_final_metrics = {}

    for budget in args.budgets:
        print(f"\n[online optimize] budget={budget}")
        final_layer_patch = run_budget_online(
            ctx=ctx,
            args=args,
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
    save_results(layer_idx_list, layer_results, budget_to_final_metrics, args)

    print("Running one-shot multi-layer baseline comparison...")
    run_multilayer_baseline_check(
        ctx=ctx,
        args=args,
        target_layers=layer_idx_list,
        head_idx=head_idx,
        pos_list=pos_list,
        model_inputs=model_inputs,
        ref_tail_logits=ref_tail_logits,
    )
