import argparse

from transformers import AutoTokenizer

from ...runner_utils import build_prompt
from .common import ALIGNED_PROTOCOL, LEGACY_PROTOCOL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that the token stream can provide all requested windows."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument(
        "--ppl-protocol",
        choices=[LEGACY_PROTOCOL, ALIGNED_PROTOCOL],
        default=LEGACY_PROTOCOL,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stride = args.seq_len if args.sample_stride is None else args.sample_stride
    if args.seq_len <= 0 or args.num_samples <= 0 or stride <= 0:
        raise ValueError("seq-len, num-samples, and sample-stride must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    join_separator = "\n\n" if args.ppl_protocol == ALIGNED_PROTOCOL else "\n"
    token_ids = tokenizer(
        build_prompt(
            args.dataset,
            join_separator=join_separator,
            filter_empty=args.ppl_protocol != ALIGNED_PROTOCOL,
        ),
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    available = len(token_ids)
    last_start = args.start_offset + (args.num_samples - 1) * stride
    target_lookahead = 0 if args.ppl_protocol == ALIGNED_PROTOCOL else 1
    required = last_start + args.seq_len + target_lookahead
    max_samples = (
        0
        if available < args.start_offset + args.seq_len + target_lookahead
        else (
            available
            - args.start_offset
            - args.seq_len
            - target_lookahead
        )
        // stride
        + 1
    )

    print(f"ppl_protocol={args.ppl_protocol}")
    print(f"available_tokens={available}")
    print(f"required_tokens={required}")
    print(f"token_margin={available - required}")
    print(f"last_window_start={last_start}")
    print(f"max_samples_for_configuration={max_samples}")
    if available < required:
        raise ValueError(
            f"Dataset is too short: available={available}, required={required}"
        )
    print("dataset_check=PASS")


if __name__ == "__main__":
    main()
