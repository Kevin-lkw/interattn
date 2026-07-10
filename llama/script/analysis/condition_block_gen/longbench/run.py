import argparse
import json
from pathlib import Path

from datasets import load_dataset

from ..common import add_generation_args, run_generation_benchmark


CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_config(name):
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


DATASET2PROMPT = load_config("dataset2prompt.json")
DATASET2MAXLEN = load_config("dataset2maxlen.json")
MODEL2PATH = load_config("model2path.json")
MODEL2MAXLEN = load_config("model2maxlen.json")

LONGBENCH_DATASETS = list(DATASET2MAXLEN)
LONGBENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def resolve_dataset_name(dataset, use_longbench_e):
    if use_longbench_e:
        if dataset not in LONGBENCH_E_DATASETS:
            raise ValueError(f"{dataset} is not available in LongBench-E.")
        return f"{dataset}_e"
    return dataset


def resolve_model_config(args):
    model_alias = args.model
    if model_alias in MODEL2PATH:
        args.model = MODEL2PATH[model_alias]
    if args.max_input_tokens is None and model_alias in MODEL2MAXLEN:
        args.max_input_tokens = int(MODEL2MAXLEN[model_alias])
    return args


def load_longbench_records(args):
    hf_name = resolve_dataset_name(args.dataset, args.longbench_e)
    data = load_dataset(
        args.hf_repo,
        hf_name,
        split=args.split,
        trust_remote_code=True,
    )
    records = [dict(item) for item in data]
    for index, record in enumerate(records):
        record.setdefault("dataset", args.dataset)
        record.setdefault("_id", record.get("id", index))
    return records


def longbench_prompt(context, question, record):
    dataset = str(record.get("dataset", "")).lower()
    if dataset.endswith("_e"):
        dataset = dataset[:-2]
    return DATASET2PROMPT[dataset].format(
        context=context,
        input=record.get("input", question),
    )


def parse_args():
    parser = add_generation_args(
        argparse.ArgumentParser(description="Run LongBench v1 generation.")
    )
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=LONGBENCH_DATASETS,
        help="LongBench v1 dataset config to load from Hugging Face.",
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
        help="Load the LongBench-E variant for datasets that provide it.",
    )
    return parser.parse_args()


def main():
    args = resolve_model_config(parse_args())
    args.question_field = "input"
    args.id_field = "_id"
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET2MAXLEN[args.dataset]
    records = None if args.data is not None else load_longbench_records(args)
    benchmark_name = f"longbench/{resolve_dataset_name(args.dataset, args.longbench_e)}"
    run_generation_benchmark(
        args,
        benchmark_name=benchmark_name,
        prompt_builder=longbench_prompt,
        records=records,
    )


if __name__ == "__main__":
    main()
