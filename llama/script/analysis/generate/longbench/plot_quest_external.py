import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .eval import DATASET2METRIC, DEFAULT_RESULT_ROOT
from .eval_dataset_plot import plot_score_vs_budget, write_csv, write_json
from .run import DATASET2PROMPT


QUEST_RESULT_JSON = Path(
    "/scratch1/liankewei/quest/evaluation/LongBench/pred/"
    "Meta-Llama-3.1-8B-Instruct/result.json"
)
QUEST_PRED_DIR = QUEST_RESULT_JSON.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert external QUEST LongBench token budgets to ratios and plot with local summaries."
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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional root for outputs. Defaults to writing merged plots under "
            "<result-root>/<model>/<benchmark>/<dataset>/eval_plots."
        ),
    )
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def coerce_row(row):
    out = dict(row)
    for key in [
        "score",
        "effective_budget",
        "effective_decode_budget",
        "mean_sample_budget",
        "budget",
        "eps",
        "mean_generated_tokens",
        "mean_input_tokens",
    ]:
        if out.get(key) in (None, ""):
            out[key] = None
        elif key in out:
            out[key] = float(out[key])
    for key in ["block_size", "num_predictions", "file_size_bytes"]:
        if out.get(key) in (None, ""):
            out[key] = None
        elif key in out:
            out[key] = int(float(out[key]))
    return out


def apply_llama3_chat_template(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def quest_context_length(tokenizer, prompt, dataset):
    if dataset in ["qasper", "hotpotqa"]:
        q_pos = prompt.rfind("Question:")
    elif dataset in ["multifieldqa_en", "gov_report"]:
        q_pos = prompt.rfind("Now,")
    elif dataset == "triviaqa":
        q_pos = prompt.rfind("Answer the question")
    elif dataset == "narrativeqa":
        q_pos = prompt.rfind("Do not provide")
    else:
        q_pos = -1

    q_pos = max(len(prompt) - 100, q_pos)
    prefix = prompt[:q_pos]
    question = prompt[q_pos:]
    prefix_ids = tokenizer(prefix, truncation=False, return_tensors="pt").input_ids[0]
    question_ids = tokenizer(question, truncation=False, return_tensors="pt").input_ids[0]
    return int(prefix_ids.shape[-1] + max(int(question_ids.shape[-1]) - 1, 0))


def load_quest_input_lengths(args, tokenizer, dataset):
    data = load_dataset(args.hf_repo, dataset, split="test")
    lengths = []
    for record in data:
        prompt = DATASET2PROMPT[dataset].format(**dict(record))
        prompt = apply_llama3_chat_template(tokenizer, prompt)
        lengths.append(quest_context_length(tokenizer, prompt, dataset))
    return lengths


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def budget_stats(token_budget, input_lengths):
    sample_budgets = [min(float(token_budget) / float(length), 1.0) for length in input_lengths]
    return {
        "effective_budget": mean(sample_budgets),
        "effective_decode_budget": mean(sample_budgets),
        "mean_sample_budget": mean(sample_budgets),
        "mean_input_tokens": mean(float(length) for length in input_lengths),
    }


def parse_quest_filename(filename):
    match = re.fullmatch(r"(.+)-full\.jsonl", filename)
    if match:
        return match.group(1), None
    match = re.fullmatch(r"(.+)-(\d+)\.jsonl", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def build_quest_rows(args):
    result = json.loads(args.quest_result_json.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
    length_cache = {}
    rows_by_dataset = {dataset: [] for dataset in args.datasets}

    for filename, score in sorted(result.items()):
        dataset, token_budget = parse_quest_filename(filename)
        if dataset not in rows_by_dataset:
            continue

        pred_path = args.quest_pred_dir / filename
        file_mtime = None
        file_size = None
        num_predictions = None
        if pred_path.exists():
            stat = pred_path.stat()
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            file_size = stat.st_size
            with pred_path.open("r", encoding="utf-8") as handle:
                num_predictions = sum(1 for line in handle if line.strip())

        if token_budget is None:
            budget_info = {
                "effective_budget": 1.0,
                "effective_decode_budget": 1.0,
                "mean_sample_budget": 1.0,
                "mean_input_tokens": None,
            }
            run = "quest_external_full"
            label = "QUEST full"
            budget = 1.0
        else:
            if dataset not in length_cache:
                length_cache[dataset] = load_quest_input_lengths(args, tokenizer, dataset)
            budget_info = budget_stats(token_budget, length_cache[dataset])
            run = f"quest_external_tokens={token_budget}"
            label = f"QUEST {token_budget} tok"
            budget = budget_info["effective_decode_budget"]

        rows_by_dataset[dataset].append(
            {
                "dataset": dataset,
                "run": run,
                "label": label,
                "method": "quest",
                "score": float(score),
                "budget": budget,
                "token_budget": token_budget,
                "num_predictions": num_predictions,
                "file_mtime_utc": file_mtime,
                "file_size_bytes": file_size,
                **budget_info,
            }
        )

    return rows_by_dataset


def write_quest_csv(path, rows):
    fields = [
        "dataset",
        "run",
        "label",
        "method",
        "score",
        "effective_budget",
        "effective_decode_budget",
        "mean_sample_budget",
        "budget",
        "token_budget",
        "num_predictions",
        "file_mtime_utc",
        "file_size_bytes",
        "mean_input_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.result_root / args.model / args.benchmark
    output_root = args.output_dir or root
    output_root.mkdir(parents=True, exist_ok=True)

    rows_by_dataset = build_quest_rows(args)
    all_quest_rows = []
    for dataset, quest_rows in rows_by_dataset.items():
        all_quest_rows.extend(quest_rows)
        dataset_dir = root / dataset
        local_rows = read_csv_rows(dataset_dir / "eval_plots" / "eval_summary.csv")
        merged_rows = local_rows + quest_rows

        if args.output_dir is None:
            dataset_output_dir = dataset_dir / "eval_plots"
        else:
            dataset_output_dir = output_root / dataset
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(dataset_output_dir / "eval_summary_with_quest.json", merged_rows)
        write_csv(dataset_output_dir / "eval_summary_with_quest.csv", merged_rows)
        plot_score_vs_budget(
            dataset_output_dir / "score_vs_effective_decode_budget_with_quest.png",
            merged_rows,
            dataset,
            args.plot_dpi,
        )
        plot_score_vs_budget(
            dataset_output_dir / "score_vs_effective_decode_budget_with_quest_log.png",
            merged_rows,
            dataset,
            args.plot_dpi,
            xscale="log",
        )
        print(f"Saved merged plot: {dataset_output_dir / 'score_vs_effective_decode_budget_with_quest.png'}")
        print(f"Saved merged plot: {dataset_output_dir / 'score_vs_effective_decode_budget_with_quest_log.png'}")

    write_quest_csv(root / "quest_external_budget_summary.csv", all_quest_rows)
    write_json(root / "quest_external_budget_summary.json", all_quest_rows)
    print(f"Saved QUEST summary: {root / 'quest_external_budget_summary.csv'}")


if __name__ == "__main__":
    main()
