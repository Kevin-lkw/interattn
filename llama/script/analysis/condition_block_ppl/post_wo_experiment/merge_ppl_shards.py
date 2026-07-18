"""Merge independently completed PPL windows into one corpus-level summary."""

import argparse
import copy
import json
from pathlib import Path

import torch

from .run_ppl import RESULT_ROOT, _aggregate, _plot, _save


CONFIG_KEYS = (
    "model",
    "dataset",
    "dtype",
    "seq_len",
    "ppl_protocol",
    "block_size",
    "eps",
    "variants",
    "delta_mode",
    "full_attention_layers",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            RESULT_ROOT / "ppl" / "summary.pt",
            RESULT_ROOT / "ppl_shards" / "start_1024" / "summary.pt",
            RESULT_ROOT / "ppl_shards" / "start_2048" / "summary.pt",
            RESULT_ROOT / "ppl_shards" / "start_3072" / "summary.pt",
            RESULT_ROOT / "ppl_shards" / "start_4096" / "summary.pt",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "ppl_n5")
    return parser.parse_args()


def _signature(config):
    return {key: config.get(key) for key in CONFIG_KEYS}


def main():
    args = parse_args()
    sources = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    if not sources:
        raise ValueError("At least one input summary is required")
    expected = _signature(sources[0]["config"])
    samples_by_start = {}
    for path, source in zip(args.inputs, sources):
        actual = _signature(source["config"])
        if actual != expected:
            raise ValueError(
                f"Incompatible config in {path}: expected={expected}, actual={actual}"
            )
        for sample in source["samples"].values():
            start = int(sample["start"])
            if start in samples_by_start:
                raise ValueError(f"Duplicate sample start {start} in {path}")
            for variant in expected["variants"]:
                missing = set(map(float, expected["eps"])) - set(
                    sample.get("results", {}).get(variant, {})
                )
                if missing:
                    raise ValueError(
                        f"Incomplete sample start={start}, variant={variant}: "
                        f"missing eps={sorted(missing)}"
                    )
            samples_by_start[start] = copy.deepcopy(sample)

    starts = sorted(samples_by_start)
    config = copy.deepcopy(sources[0]["config"])
    config.update(
        {
            "device": "multi_gpu",
            "num_samples": len(starts),
            "start_offset": starts[0],
            "sample_stride": None,
            "output_dir": str(args.output_dir),
            "output_root": str(args.output_dir),
            "source_summaries": [str(path) for path in args.inputs],
        }
    )
    summary = {
        "config": config,
        "starts": starts,
        "samples": {
            sample_idx: samples_by_start[start]
            for sample_idx, start in enumerate(starts)
        },
        "aggregate": {},
    }
    summary["aggregate"] = _aggregate(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save(summary, args.output_dir / "summary.pt")
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(summary["aggregate"], indent=2) + "\n"
    )
    _plot(summary, args.output_dir / "ppl_vs_budget.png")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Merged {len(starts)} windows with starts={starts}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
