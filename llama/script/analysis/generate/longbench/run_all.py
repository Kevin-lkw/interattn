import argparse
import json
import traceback
from copy import copy
from functools import lru_cache
from pathlib import Path

import torch
from tqdm import tqdm

from ..common import (
    add_generation_args,
    load_done_ids,
    load_model_and_tokenizer,
    output_path,
    pending_generation_records,
    run_generation_benchmark,
    validate_generation_args,
)
from .run import (
    DATASET2MAXLEN,
    LONGBENCH_DATASETS,
    load_longbench_records,
    longbench_prompt,
    resolve_dataset_name,
    resolve_model_config,
)


LONGBENCH_EN_CODE_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


TASK_INFO_PATH = Path(__file__).resolve().with_name("task_info.json")
TASK_INFO_E_PATH = Path(__file__).resolve().with_name("task_info_e.json")


@lru_cache(maxsize=None)
def _load_task_counts(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts = {}
    for dataset, info in payload.items():
        if "num_test" in info:
            counts[dataset] = int(info["num_test"])
    return counts


def _expected_record_count(args, benchmark_name):
    if args.data is not None:
        return None
    task_info_path = TASK_INFO_E_PATH if args.longbench_e else TASK_INFO_PATH
    count = _load_task_counts(task_info_path).get(benchmark_name.removeprefix("longbench/"))
    if count is None:
        return None
    if args.limit is not None:
        return min(count, args.limit)
    return count


def _fast_skip_complete(args, benchmark_name):
    expected_count = _expected_record_count(args, benchmark_name)
    if expected_count is None:
        return None
    out_path = output_path(args, benchmark_name)
    done_count = len(load_done_ids(out_path))
    if done_count >= expected_count:
        return out_path, done_count, expected_count
    return None


def parse_args():
    parser = add_generation_args(
        argparse.ArgumentParser(
            description="Run one compression method on 14 English LongBench tasks plus 2 code tasks."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=LONGBENCH_EN_CODE_DATASETS,
        choices=LONGBENCH_DATASETS,
        help="LongBench dataset configs to run.",
    )
    parser.add_argument(
        "--hf-repo",
        default="THUDM/LongBench",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--longbench-e",
        action="store_true",
        help="Load LongBench-E variants where available.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log dataset failures and continue with the remaining datasets.",
    )
    parser.add_argument(
        "--disable-fast-skip",
        action="store_true",
        help=(
            "Always load dataset records before checking for completed predictions. "
            "By default, known LongBench task counts are used to skip complete outputs."
        ),
    )
    return parser.parse_args()


def main():
    args = resolve_model_config(parse_args())
    requested_max_new_tokens = args.max_new_tokens
    args.question_field = "input"
    args.id_field = "_id"
    validate_generation_args(copy(args))

    pending_jobs = []
    for dataset in tqdm(args.datasets, desc="LongBench datasets", unit="dataset"):
        dataset_args = copy(args)
        dataset_args.dataset = dataset
        dataset_args.max_new_tokens = requested_max_new_tokens or DATASET2MAXLEN[dataset]
        benchmark_name = f"longbench/{resolve_dataset_name(dataset, dataset_args.longbench_e)}"
        try:
            if not dataset_args.disable_fast_skip:
                fast_skip = _fast_skip_complete(dataset_args, benchmark_name)
                if fast_skip is not None:
                    out_path, done_count, expected_count = fast_skip
                    tqdm.write(
                        "All predictions already exist; fast-skipping "
                        f"{benchmark_name} ({done_count}/{expected_count} ids): {out_path}"
                    )
                    continue
            records = None if dataset_args.data is not None else load_longbench_records(dataset_args)
            records, out_path, _done_ids, pending_records = pending_generation_records(
                dataset_args,
                benchmark_name=benchmark_name,
                records=records,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            tqdm.write(f"Failed {benchmark_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=20)
            continue
        if not pending_records:
            tqdm.write(f"All predictions already exist; skipping {benchmark_name}: {out_path}")
            continue
        pending_jobs.append((dataset_args, benchmark_name, records))

    if not pending_jobs:
        print("All requested predictions already exist; skipping model load.")
        return

    model, tokenizer = load_model_and_tokenizer(args)
    for dataset_args, benchmark_name, records in tqdm(
        pending_jobs,
        desc="LongBench pending datasets",
        unit="dataset",
    ):
        try:
            run_generation_benchmark(
                dataset_args,
                benchmark_name=benchmark_name,
                prompt_builder=longbench_prompt,
                records=records,
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            tqdm.write(f"Failed {benchmark_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=20)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
