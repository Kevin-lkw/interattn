import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..common import build_prompt as build_generation_prompt
from .run import (
    DATASET2MAXLEN,
    LONGBENCH_DATASETS,
    LONGBENCH_E_DATASETS,
    longbench_prompt,
    resolve_dataset_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print LongBench task metadata as JSON."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=LONGBENCH_DATASETS,
        help="LongBench dataset configs to include. Defaults to all LongBench tasks.",
    )
    parser.add_argument(
        "--hf-repo",
        default="THUDM/LongBench",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to task_info.json in this longbench folder.",
    )
    parser.add_argument(
        "--longbench-e",
        action="store_true",
        help="Print LongBench-E task info for datasets that provide it.",
    )
    return parser.parse_args()


def count_jsonl_member(archive, member):
    with archive.open(member) as handle:
        return sum(1 for line in handle if line.strip())


def load_jsonl_member(archive, member, dataset):
    records = []
    with archive.open(member) as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("dataset", dataset)
            record.setdefault("_id", record.get("id", index))
            records.append(record)
    return records


def input_length_stats(records):
    lengths = []
    prompt_args = argparse.Namespace(
        context_field="context",
        question_field="input",
        prompt_field=None,
    )
    for record in records:
        prompt = build_generation_prompt(record, prompt_args, longbench_prompt)
        lengths.append(len(prompt))
    return {
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
    }


def build_task_info(args):
    datasets = args.datasets
    if datasets is None:
        datasets = LONGBENCH_E_DATASETS if args.longbench_e else LONGBENCH_DATASETS

    zip_path = hf_hub_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        filename="data.zip",
    )

    info = {}
    with zipfile.ZipFile(zip_path) as archive:
        for dataset in datasets:
            hf_name = resolve_dataset_name(dataset, args.longbench_e)
            member = f"data/{hf_name}.jsonl"
            records = load_jsonl_member(archive, member, dataset)
            info[hf_name] = {
                "dataset": dataset,
                "num_test": len(records),
                "max_new_tokens": DATASET2MAXLEN[dataset],
                "input_length_chars": input_length_stats(records),
            }
    return info


def main():
    args = parse_args()
    output = args.output
    if output is None:
        filename = "task_info_e.json" if args.longbench_e else "task_info.json"
        output = Path(__file__).resolve().with_name(filename)
    info = build_task_info(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved task info to: {output}")


if __name__ == "__main__":
    main()
