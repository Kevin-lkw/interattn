import argparse
import math
import random

import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.config import set_seed, str_to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample fixed-length windows from WikiText and print the distribution of "
            "per-sequence perplexity under full causal attention."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--num-sequences", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument("--hist-width", type=int, default=50)
    return parser.parse_args()


def validate_args(args):
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be > 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.hist_bins <= 0:
        raise ValueError("--hist-bins must be > 0")
    if args.hist_width <= 0:
        raise ValueError("--hist-width must be > 0")


def load_wikitext_tokens(tokenizer, split):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [
        text
        for text in dataset["text"]
        if isinstance(text, str) and text.strip()
    ]
    if not texts:
        raise ValueError(f"No non-empty text found in WikiText split '{split}'")

    encoded = tokenizer(
        "\n".join(texts),
        add_special_tokens=True,
        return_attention_mask=False,
        return_tensors="pt",
    )
    return encoded["input_ids"][0]


def sample_window_starts(num_tokens, seq_len, num_sequences, seed):
    # Each window needs one extra token as the target of its final input token.
    num_starts = num_tokens - seq_len
    if num_starts <= 0:
        raise ValueError(
            f"Token stream has {num_tokens} tokens, but at least {seq_len + 1} are required"
        )
    if num_sequences > num_starts:
        raise ValueError(
            f"Cannot sample {num_sequences} unique starts from only {num_starts} positions"
        )

    rng = random.Random(seed)
    return rng.sample(range(num_starts), num_sequences)


def sequence_nlls(model, token_ids, starts, seq_len, batch_size, device):
    all_nlls = []
    total_nll = 0.0
    total_tokens = 0

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        windows = torch.stack(
            [token_ids[start : start + seq_len + 1] for start in batch_starts]
        )
        input_ids = windows[:, :-1].to(device)
        labels = windows[:, 1:].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits
            token_nlls = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(len(batch_starts), seq_len)
            batch_nlls = token_nlls.mean(dim=1)

        batch_nlls_cpu = batch_nlls.cpu()
        all_nlls.extend(batch_nlls_cpu.tolist())
        total_nll += float(token_nlls.sum().item())
        total_tokens += token_nlls.numel()

        for offset, (start, nll) in enumerate(zip(batch_starts, batch_nlls_cpu)):
            index = batch_start + offset
            print(
                f"sequence={index:03d} start={start:7d} "
                f"nll={float(nll):.6f} ppl={math.exp(float(nll)):.6f}"
            )

    return torch.tensor(all_nlls, dtype=torch.float64), total_nll / total_tokens


def print_histogram(values, bins, width):
    values = values.float()
    low = float(values.min())
    high = float(values.max())
    if low == high:
        print(f"[{low:.4f}, {high:.4f}] " + "#" * width + f" {values.numel()}")
        return

    counts = torch.histc(values, bins=bins, min=low, max=high).to(torch.int64)
    max_count = max(int(counts.max()), 1)
    edges = torch.linspace(low, high, bins + 1)
    for index, count in enumerate(counts.tolist()):
        bar_len = round(width * count / max_count)
        right_bracket = "]" if index == bins - 1 else ")"
        print(
            f"[{float(edges[index]):9.4f}, {float(edges[index + 1]):9.4f}"
            f"{right_bracket} {'#' * bar_len:<{width}} {count:4d}"
        )


def print_summary(nlls, corpus_nll, hist_bins, hist_width):
    ppls = nlls.exp()
    quantile_levels = torch.tensor(
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
        dtype=ppls.dtype,
    )
    quantiles = torch.quantile(ppls, quantile_levels)

    print("\nPPL summary")
    print(f"num_sequences={ppls.numel()}")
    print(f"corpus_nll={corpus_nll:.6f}")
    print(f"corpus_ppl={math.exp(corpus_nll):.6f}")
    print(f"sequence_ppl_mean={float(ppls.mean()):.6f}")
    print(f"sequence_ppl_std={float(ppls.std(unbiased=False)):.6f}")
    for level, value in zip(quantile_levels.tolist(), quantiles.tolist()):
        print(f"sequence_ppl_p{round(level * 100):02d}={value:.6f}")

    print("\nSequence PPL histogram")
    print_histogram(ppls, bins=hist_bins, width=hist_width)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    dtype = str_to_torch_dtype(args.dtype)

    print(f"Loading tokenizer and WikiText-2 {args.split} split...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    token_ids = load_wikitext_tokens(tokenizer, args.split)
    starts = sample_window_starts(
        num_tokens=token_ids.numel(),
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        seed=args.seed,
    )

    print(
        f"tokens={token_ids.numel()}, sequences={args.num_sequences}, "
        f"seq_len={args.seq_len}, seed={args.seed}"
    )
    print("Loading model with eager full attention...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and args.seq_len > max_positions:
        raise ValueError(
            f"--seq-len {args.seq_len} exceeds model max_position_embeddings "
            f"({max_positions})"
        )

    nlls, corpus_nll = sequence_nlls(
        model=model,
        token_ids=token_ids,
        starts=starts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    print_summary(nlls, corpus_nll, args.hist_bins, args.hist_width)


if __name__ == "__main__":
    main()
