import math

import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import parse_args, str_to_torch_dtype


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
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported now: wikitext, pg19, oasst2"
        )

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

    input_ids = encoded["input_ids"][:, args.start : args.start + args.seq_len].to(args.device)
    labels = encoded["input_ids"][:, args.start + 1 : args.start + args.seq_len + 1].to(args.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    ).item()
    ppl = math.exp(nll)

    print(f"dataset={args.dataset}")
    print(f"model={args.model}")
    print(f"start={args.start}, seq_len={args.seq_len}")
    print(f"nll={nll:.6f}")
    print(f"ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
