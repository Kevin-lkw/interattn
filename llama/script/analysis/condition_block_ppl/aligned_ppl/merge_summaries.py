import argparse
import os
from pathlib import Path

import torch

from ..multisample.common import aggregate_samples, plot_aggregate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge disjoint aligned-PPL sample shards by token start."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--setting-label", default="setting")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    methods = {summary["method"] for summary in summaries}
    settings = {tuple(summary["settings"]) for summary in summaries}
    if len(methods) != 1 or len(settings) != 1:
        raise ValueError("Shard methods and settings must match.")

    samples_by_start = {}
    for summary in summaries:
        for sample in summary["samples"].values():
            start = int(sample["start"])
            if start in samples_by_start:
                raise ValueError(f"Duplicate sample at start={start}.")
            samples_by_start[start] = sample

    starts = sorted(samples_by_start)
    if len(starts) != args.expected_samples:
        raise ValueError(
            f"Found {len(starts)} unique samples, expected {args.expected_samples}."
        )
    if len(starts) > 1:
        stride = starts[1] - starts[0]
        expected_starts = [starts[0] + index * stride for index in range(len(starts))]
        if starts != expected_starts:
            raise ValueError("Merged samples are not a regular contiguous stride.")
    else:
        stride = int(summaries[0]["config"]["sample_stride"])

    merged = {
        "method": summaries[0]["method"],
        "config": dict(summaries[0]["config"]),
        "starts": starts,
        "settings": list(summaries[0]["settings"]),
        "samples": {
            index: samples_by_start[start] for index, start in enumerate(starts)
        },
        "aggregate": {},
        "merged_from": [str(path) for path in args.inputs],
    }
    merged["config"]["num_samples"] = len(starts)
    merged["config"]["start_offset"] = starts[0]
    merged["config"]["sample_stride"] = stride
    merged["config"]["output_root"] = args.output.parent.parent
    merged["aggregate"] = aggregate_samples(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(merged, tmp_path)
    os.replace(tmp_path, args.output)
    print(
        f"Merged {len(starts)} samples for {merged['method']} into {args.output}"
    )
    if args.plot is not None:
        plot_aggregate(merged, args.plot, args.setting_label)
        print(f"Saved merged plot to {args.plot}")


if __name__ == "__main__":
    main()
