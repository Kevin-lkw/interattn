import argparse
import json
import re
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from .metric import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)


DEFAULT_RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "generate"

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate local LongBench generations.")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument("--e", action="store_true", help="Evaluate LongBench-E predictions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to eval.json under the model benchmark folder.",
    )
    return parser.parse_args(args)


def load_metadata(hf_repo, dataset, use_longbench_e):
    hf_name = f"{dataset}_e" if use_longbench_e else dataset
    zip_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename="data.zip")
    metadata = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"data/{hf_name}.jsonl") as handle:
            for index, raw_line in enumerate(handle):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                record = json.loads(line)
                record_id = str(record.get("_id", record.get("id", index)))
                metadata[record_id] = {
                    "all_classes": record.get("all_classes"),
                    "length": record.get("length"),
                }
    return metadata


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_rows(dataset, rows, metadata, use_longbench_e):
    if use_longbench_e:
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
    else:
        scores = []
    metric = DATASET2METRIC[dataset]

    for row in rows:
        prediction = str(row.get("prediction", row.get("pred", "")))
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]

        record_meta = metadata.get(str(row.get("id", "")), {})
        all_classes = row.get("all_classes", record_meta.get("all_classes"))
        answers = row.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers]

        sample_score = 0.0
        for answer in answers:
            sample_score = max(
                sample_score,
                metric(prediction, str(answer), all_classes=all_classes),
            )

        if use_longbench_e:
            length = row.get("length", record_meta.get("length"))
            if length is None:
                raise ValueError(f"Missing length for LongBench-E row id={row.get('id')}")
            if length < 4000:
                scores["0-4k"].append(sample_score)
            elif length < 8000:
                scores["4-8k"].append(sample_score)
            else:
                scores["8k+"].append(sample_score)
        else:
            scores.append(sample_score)

    if use_longbench_e:
        return {
            key: round(100 * sum(values) / len(values), 2) if values else None
            for key, values in scores.items()
        }
    return round(100 * sum(scores) / len(scores), 2) if scores else None


def parse_run_name(path):
    match = re.fullmatch(r"(.+)_budget=([^_]+)_maxnew=(\d+)", path.stem)
    if match is None:
        return {"run": path.stem}
    method, budget, max_new_tokens = match.groups()
    return {
        "run": path.stem,
        "method": method,
        "budget": float(budget),
        "max_new_tokens": int(max_new_tokens),
    }


def evaluate(args):
    root = args.result_root / args.model / args.benchmark
    if not root.exists():
        raise FileNotFoundError(f"Result folder does not exist: {root}")

    results = {}
    metadata_cache = {}
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name[:-2] if dataset_dir.name.endswith("_e") else dataset_dir.name
        if dataset not in DATASET2METRIC:
            continue
        if dataset not in metadata_cache:
            metadata_cache[dataset] = load_metadata(args.hf_repo, dataset, args.e)

        dataset_scores = {}
        for pred_path in sorted(dataset_dir.glob("*.jsonl")):
            rows = read_jsonl(pred_path)
            run_info = parse_run_name(pred_path)
            run_info["num_predictions"] = len(rows)
            run_info["score"] = score_rows(
                dataset,
                rows,
                metadata_cache[dataset],
                args.e,
            )
            dataset_scores[pred_path.stem] = run_info
        if dataset_scores:
            results[dataset_dir.name] = dataset_scores
    return results


def main():
    args = parse_args()
    output = args.output
    if output is None:
        output = args.result_root / args.model / args.benchmark / "eval.json"
    results = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved evaluation to: {output}")


if __name__ == "__main__":
    main()
