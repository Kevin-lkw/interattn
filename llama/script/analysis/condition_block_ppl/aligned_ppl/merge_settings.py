import argparse
import math
import os
from pathlib import Path

import torch

from ..multisample.common import aggregate_samples, plot_aggregate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge full-coverage summaries for disjoint setting grids while "
            "requiring identical samples and teacher losses."
        )
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--setting-label", default="setting")
    return parser.parse_args()


def _samples_by_start(summary):
    samples = {}
    for sample in summary["samples"].values():
        start = int(sample["start"])
        if start in samples:
            raise ValueError(f"Duplicate sample at start={start} within one summary.")
        samples[start] = sample
    return samples


def _merge_method_config(config, extension_config):
    merged = dict(config)
    for key in ("budgets", "eps"):
        if key in config and key in extension_config:
            merged[key] = sorted(
                {float(value) for value in config[key]}
                | {float(value) for value in extension_config[key]}
            )
    if "p_settings" in config and "p_settings" in extension_config:
        p_settings = {
            float(p1): (float(p1), float(p2))
            for p1, p2 in config["p_settings"]
        }
        for p1, p2 in extension_config["p_settings"]:
            p1 = float(p1)
            p2 = float(p2)
            if p1 in p_settings and p_settings[p1][1] != p2:
                raise ValueError(f"Conflicting p2 values for p1={p1:g}.")
            p_settings[p1] = (p1, p2)
        merged["p_settings"] = [p_settings[p1] for p1 in sorted(p_settings)]
        merged["p2_by_p1"] = {
            p1: p2 for p1, p2 in merged["p_settings"]
        }
    return merged


def merge_setting_summaries(summaries, input_paths=None):
    if not summaries:
        raise ValueError("At least one summary is required.")
    methods = {summary["method"] for summary in summaries}
    if len(methods) != 1:
        raise ValueError(f"Methods do not match: {sorted(methods)}")

    base = summaries[0]
    starts = [int(start) for start in base["starts"]]
    base_samples = _samples_by_start(base)
    if sorted(base_samples) != sorted(starts):
        raise ValueError("Base summary is not complete for its declared starts.")

    combined_settings = []
    seen_settings = set()
    config = dict(base["config"])
    samples = {
        start: {
            **sample,
            "results": dict(sample["results"]),
        }
        for start, sample in base_samples.items()
    }
    for setting in base["settings"]:
        setting = float(setting)
        combined_settings.append(setting)
        seen_settings.add(setting)

    for summary in summaries[1:]:
        if [int(start) for start in summary["starts"]] != starts:
            raise ValueError("Summaries do not declare the same ordered starts.")
        for key in ("ppl_protocol", "seq_len", "dtype", "model"):
            if summary["config"].get(key) != base["config"].get(key):
                raise ValueError(f"Config mismatch for {key!r}.")
        extension_samples = _samples_by_start(summary)
        if set(extension_samples) != set(samples):
            raise ValueError("Summaries do not contain the same completed starts.")
        extension_settings = [float(setting) for setting in summary["settings"]]
        overlap = seen_settings.intersection(extension_settings)
        if overlap:
            raise ValueError(f"Setting grids overlap: {sorted(overlap)}")

        for start in starts:
            base_sample = samples[start]
            extension_sample = extension_samples[start]
            if int(base_sample["num_tokens"]) != int(extension_sample["num_tokens"]):
                raise ValueError(f"Token-count mismatch at start={start}.")
            if not math.isclose(
                float(base_sample["teacher_nll"]),
                float(extension_sample["teacher_nll"]),
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(f"Teacher-NLL mismatch at start={start}.")
            extension_results = {
                float(setting): record
                for setting, record in extension_sample["results"].items()
            }
            if set(extension_results) != set(extension_settings):
                raise ValueError(
                    f"Incomplete extension results at start={start}."
                )
            base_sample["results"].update(extension_results)

        combined_settings.extend(extension_settings)
        seen_settings.update(extension_settings)
        config = _merge_method_config(config, summary["config"])

    combined_settings.sort()
    merged = {
        "method": base["method"],
        "config": config,
        "starts": starts,
        "settings": combined_settings,
        "samples": {
            index: samples[start] for index, start in enumerate(starts)
        },
        "aggregate": {},
    }
    if input_paths is not None:
        merged["settings_merged_from"] = [str(path) for path in input_paths]
    merged["aggregate"] = aggregate_samples(merged)
    return merged


def main():
    args = parse_args()
    summaries = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    merged = merge_setting_summaries(summaries, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(merged, tmp_path)
    os.replace(tmp_path, args.output)
    print(
        f"Merged {len(merged['settings'])} settings for {merged['method']} "
        f"into {args.output}"
    )
    if args.plot is not None:
        plot_aggregate(merged, args.plot, args.setting_label)
        print(f"Saved merged plot to {args.plot}")


if __name__ == "__main__":
    main()
