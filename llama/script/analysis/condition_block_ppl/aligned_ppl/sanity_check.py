import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ...sanity import move_model_inputs_to_device
from ..multisample.common import (
    ALIGNED_PROTOCOL,
    DEFAULT_MODEL,
    DEFAULT_RESULT_ROOT,
    build_sample_context,
    load_model_and_tokens,
    metric_record,
    model_output_name,
    prepare_sample,
    validate_common_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check the aligned fixed-chunk PPL protocol."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    output = cli_args.output
    if output is None:
        output = (
            DEFAULT_RESULT_ROOT
            / model_output_name(cli_args.model)
            / "aligned_ppl"
            / "sanity"
            / "sanity.json"
        )

    args = SimpleNamespace(
        model=cli_args.model,
        dataset=cli_args.dataset,
        device=cli_args.device,
        dtype=cli_args.dtype,
        seq_len=cli_args.seq_len,
        num_samples=2,
        start_offset=0,
        sample_stride=cli_args.seq_len,
        seed=42,
        ppl_protocol=ALIGNED_PROTOCOL,
        output_root=output.parents[1],
    )
    validate_common_args(args)
    model, tokenizer, encoded, dtype, starts = load_model_and_tokens(args)

    contexts = [
        build_sample_context(
            model=model,
            tokenizer=tokenizer,
            encoded=encoded,
            dtype=dtype,
            device=args.device,
            start=start,
            seq_len=args.seq_len,
            ppl_protocol=ALIGNED_PROTOCOL,
        )
        for start in starts
    ]
    ctx = contexts[0]
    pos_list, model_inputs, labels, ref_logits, teacher_nll, teacher_ppl = (
        prepare_sample(ctx, args.seq_len)
    )

    expected_labels = model_inputs["input_ids"][:, 1:]
    if not torch.equal(labels, expected_labels):
        raise AssertionError("Aligned labels are not the within-window one-token shift.")
    if len(pos_list) != args.seq_len - 1:
        raise AssertionError("Aligned protocol must score exactly seq_len - 1 tokens.")
    if ref_logits.shape[1] != args.seq_len - 1:
        raise AssertionError("Aligned logits include a position outside the shifted loss.")

    next_ctx_inputs = contexts[1].inputs["input_ids"]
    expected_next_inputs = encoded["input_ids"][:, args.seq_len : 2 * args.seq_len]
    if not torch.equal(next_ctx_inputs, expected_next_inputs):
        raise AssertionError("Adjacent fixed-length chunks do not use the expected stride.")
    first_chunk_labels = encoded["input_ids"][:, 1 : args.seq_len]
    if not torch.equal(labels.cpu(), first_chunk_labels):
        raise AssertionError("The first chunk reads a target from the next chunk.")

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            labels=model_inputs["input_ids"],
            use_cache=False,
        )
    hf_loss = float(outputs.loss.float().item())
    manual_loss = float(
        F.cross_entropy(
            outputs.logits[:, :-1, :].float().reshape(-1, outputs.logits.size(-1)),
            model_inputs["input_ids"][:, 1:].reshape(-1),
            reduction="mean",
        ).item()
    )
    if not math.isclose(hf_loss, manual_loss, rel_tol=0.0, abs_tol=1e-5):
        raise AssertionError(
            f"HF loss ({hf_loss}) and manual shifted loss ({manual_loss}) differ."
        )
    if not math.isclose(teacher_nll, manual_loss, rel_tol=0.0, abs_tol=1e-5):
        raise AssertionError(
            f"Runner NLL ({teacher_nll}) and manual shifted loss ({manual_loss}) differ."
        )

    dense = metric_record(ref_logits, ref_logits, labels, measured_budget=1.0)
    if abs(float(dense["sanity_kl"])) > 1e-8:
        raise AssertionError("Dense self-comparison KL is not zero.")
    if abs(float(dense["nll_gap"])) > 1e-8:
        raise AssertionError("Dense self-comparison NLL gap is not zero.")

    report = {
        "status": "PASS",
        "model": args.model,
        "dataset": args.dataset,
        "ppl_protocol": args.ppl_protocol,
        "join_separator": "\\n\\n",
        "empty_rows": "preserved",
        "seq_len": args.seq_len,
        "scored_tokens_per_chunk": int(labels.numel()),
        "sample_starts": starts,
        "runner_nll": teacher_nll,
        "runner_ppl": teacher_ppl,
        "hf_shifted_loss": hf_loss,
        "manual_shifted_loss": manual_loss,
        "dense_self_kl": float(dense["sanity_kl"]),
        "dense_self_nll_gap": float(dense["nll_gap"]),
        "checks": [
            "double-newline WikiText-2 construction",
            "raw text column preserved including empty rows",
            "non-overlapping fixed-length chunks",
            "within-window next-token shift",
            "no cross-window target",
            "seq_len - 1 scored tokens",
            "HF labels loss equals manual shifted cross entropy",
            "dense self-comparison has zero KL and NLL gap",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved sanity report to: {output}")


if __name__ == "__main__":
    main()
