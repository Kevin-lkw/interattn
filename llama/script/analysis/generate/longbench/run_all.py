import argparse
import traceback
from copy import copy

import torch
from tqdm import tqdm

from ..common import (
    add_generation_args,
    load_model_and_tokenizer,
    run_generation_benchmark,
    validate_generation_args,
)
from .run import (
    DATASET2MAXLEN,
    LONGBENCH_DATASETS,
    load_longbench_records,
    longbench_prompt,
    resolve_dataset_name,
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
    return parser.parse_args()


def main():
    args = parse_args()
    requested_max_new_tokens = args.max_new_tokens
    args.question_field = "input"
    args.id_field = "_id"
    validate_generation_args(copy(args))

    model, tokenizer = load_model_and_tokenizer(args)
    for dataset in tqdm(args.datasets, desc="LongBench datasets", unit="dataset"):
        dataset_args = copy(args)
        dataset_args.dataset = dataset
        dataset_args.max_new_tokens = requested_max_new_tokens or DATASET2MAXLEN[dataset]

        records = None if dataset_args.data is not None else load_longbench_records(dataset_args)
        benchmark_name = f"longbench/{resolve_dataset_name(dataset, dataset_args.longbench_e)}"
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
