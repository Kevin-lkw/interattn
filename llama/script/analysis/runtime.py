"""Shared model/context loading helpers for analysis entry points."""

from transformers import AutoModelForCausalLM, AutoTokenizer

from .context import RunContext
from .runner_utils import build_prompt


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
    if all_layers or layer_indices is None:
        return list(range(num_hidden_layers))

    layers = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {num_hidden_layers - 1}]"
            )
        layers.append(layer_idx)

    return layers
