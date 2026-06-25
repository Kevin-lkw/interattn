import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .eval import DATASET2METRIC, DEFAULT_RESULT_ROOT
from .plot_quest_external import (
    QUEST_PRED_DIR,
    QUEST_RESULT_JSON,
    apply_llama3_chat_template,
    budget_stats,
    parse_quest_filename,
    quest_context_length,
)
from .run import DATASET2PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import external QUEST LongBench predictions into the local result tree."
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--quest-result-json", type=Path, default=QUEST_RESULT_JSON)
    parser.add_argument("--quest-pred-dir", type=Path, default=QUEST_PRED_DIR)
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"],
        choices=sorted(DATASET2METRIC),
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also import QUEST full predictions as quest_budget=1.jsonl.",
    )
    return parser.parse_args()


def output_budget_name(budget):
    return f"{float(budget):.6g}"


def load_records_and_lengths(args, tokenizer, dataset):
    data = load_dataset(args.hf_repo, dataset, split="test")
    records = []
    lengths = []
    for index, raw_record in enumerate(data):
        record = dict(raw_record)
        record_id = str(record.get("_id", record.get("id", index)))
        prompt = DATASET2PROMPT[dataset].format(**record)
        prompt = apply_llama3_chat_template(tokenizer, prompt)
        records.append(record_id)
        lengths.append(quest_context_length(tokenizer, prompt, dataset))
    return records, lengths


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def imported_rows(source_rows, record_ids, input_lengths, token_budget, run_budget):
    if len(source_rows) != len(record_ids):
        raise ValueError(
            f"Row count mismatch: source={len(source_rows)} metadata={len(record_ids)}"
        )
    rows = []
    for row, record_id, input_tokens in zip(source_rows, record_ids, input_lengths):
        sample_budget = 1.0 if token_budget is None else min(float(token_budget) / float(input_tokens), 1.0)
        rows.append(
            {
                **row,
                "id": record_id,
                "method": "quest",
                "budget": run_budget,
                "token_budget": token_budget,
                "sample_budget": sample_budget,
                "input_tokens": input_tokens,
            }
        )
    return rows


def remove_previous_imports(dataset_dir):
    for path in dataset_dir.glob("quest_budget=*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = next((line for line in handle if line.strip()), None)
            if first and "token_budget" in json.loads(first):
                path.unlink()
        except (OSError, json.JSONDecodeError):
            continue


def main():
    args = parse_args()
    root = args.result_root / args.model / args.benchmark
    result = json.loads(args.quest_result_json.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
    metadata_cache = {}
    imported = []

    for dataset in args.datasets:
        dataset_dir = root / dataset
        remove_previous_imports(dataset_dir)

    for filename in sorted(result):
        dataset, token_budget = parse_quest_filename(filename)
        if dataset not in args.datasets:
            continue
        if token_budget is None and not args.include_full:
            continue

        source_path = args.quest_pred_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing QUEST prediction file: {source_path}")

        if dataset not in metadata_cache:
            metadata_cache[dataset] = load_records_and_lengths(args, tokenizer, dataset)
        record_ids, input_lengths = metadata_cache[dataset]

        if token_budget is None:
            run_budget = 1.0
        else:
            run_budget = budget_stats(token_budget, input_lengths)["effective_decode_budget"]

        destination = (
            root
            / dataset
            / f"quest_budget={output_budget_name(run_budget)}.jsonl"
        )
        rows = imported_rows(
            read_jsonl(source_path),
            record_ids,
            input_lengths,
            token_budget,
            run_budget,
        )
        write_jsonl(destination, rows)
        imported.append((dataset, filename, destination, run_budget, len(rows)))

    for dataset, filename, destination, run_budget, count in imported:
        print(
            f"{dataset}: {filename} -> {destination} "
            f"(budget={run_budget:.6g}, rows={count})"
        )
    print(f"Imported {len(imported)} QUEST files into: {root}")


if __name__ == "__main__":
    main()
