import argparse
import os

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one forward pass and plot per-layer hidden max/mean over hidden dimension."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model path, e.g. meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "pg19", "oasst2"],
        help="Dataset used to construct one long prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model loading dtype",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start token offset in the long prompt",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for the forward pass",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=31,
        help="How many transformer layers to plot from layer 0 upward",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../result/resnorm/resnorm_layers.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-12,
        help="Small epsilon to keep y values positive for log scale",
    )
    return parser.parse_args()


def build_prompt(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip()]
    elif dataset_name == "pg19":
        dataset = load_dataset("emozilla/pg19-test", split="test")
        texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip()]
    elif dataset_name == "oasst2":
        dataset = load_dataset("OpenAssistant/oasst2", split="train")
        texts = dataset["text"]
        langs = dataset["lang"] if "lang" in dataset.column_names else None

        if langs is None:
            texts = [text for text in texts if isinstance(text, str) and text.strip()]
        else:
            texts = [
                text
                for text, lang in zip(texts, langs)
                if isinstance(text, str)
                and text.strip()
                and (lang is None or str(lang).startswith("en"))
            ]
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    if len(texts) == 0:
        raise ValueError(f"No valid text found in dataset '{dataset_name}'.")
    return "\n".join(texts)


def main():
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    prompt = build_prompt(args.dataset)
    encoded = tokenizer(
        prompt,
        max_length=args.start + args.seq_len,
        truncation=True,
        return_tensors="pt",
    )

    total_len = encoded["input_ids"].shape[1]
    required_len = args.start + args.seq_len
    if total_len < required_len:
        raise ValueError(
            f"Tokenized prompt length ({total_len}) is shorter than required ({required_len})."
        )

    input_ids = encoded["input_ids"][:, args.start : args.start + args.seq_len].to(args.device)
    attention_mask = None
    if "attention_mask" in encoded:
        attention_mask = encoded["attention_mask"][:, args.start : args.start + args.seq_len].to(
            args.device
        )

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )

    # hidden_states[0] is embedding output; hidden_states[1:] are per-layer outputs.
    hidden_states = outputs.hidden_states[1:]
    total_layers = len(hidden_states)
    num_layers = min(args.num_layers, total_layers)
    if num_layers <= 0:
        raise ValueError("--num-layers must be > 0")

    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=1,
        figsize=(12, max(2.2 * num_layers, 8)),
        sharex=True,
    )
    if num_layers == 1:
        axes = [axes]

    token_idx = torch.arange(args.seq_len).cpu().numpy()
    for layer_idx in range(num_layers):
        h = hidden_states[layer_idx].float().squeeze(0)  # [n, d]
        h_max = h.max(dim=-1).values  # max over d -> [n]
        h_mean = h.mean(dim=-1)  # mean over d -> [n]

        token_norm = h.norm(p=2, dim=-1)
        topk = min(3, token_norm.numel())
        top_vals, top_pos = torch.topk(token_norm, k=topk, largest=True)
        top_pos_list = top_pos.cpu().tolist()
        top_vals_list = [float(v) for v in top_vals.cpu().tolist()]
        top_text = ", ".join(
            [f"(n={pos}, norm={val:.6f})" for pos, val in zip(top_pos_list, top_vals_list)]
        )
        print(f"Layer {layer_idx:02d} norm top{topk}: {top_text}")

        h_max_plot = h_max.abs().clamp_min(args.log_eps).cpu().numpy()
        h_mean_plot = h_mean.abs().clamp_min(args.log_eps).cpu().numpy()

        ax = axes[layer_idx]
        ax.plot(token_idx, h_max_plot, label="|max over d|", linewidth=1.1)
        ax.plot(token_idx, h_mean_plot, label="|mean over d|", linewidth=1.1)
        ax.set_ylabel(f"L{layer_idx}")
        ax.grid(alpha=0.25)
        if layer_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("n (token index)")
    fig.suptitle(
        f"Hidden stats by layer | model={args.model} | dataset={args.dataset} | seq_len={args.seq_len}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.995])

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    print(f"Total transformer layers in model: {total_layers}")
    print(f"Plotted layers: {num_layers}")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()