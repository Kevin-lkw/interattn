import argparse
import json
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..common import add_generation_args, run_generation_benchmark


LONGBENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

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

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


def resolve_dataset_name(dataset, use_longbench_e):
    if use_longbench_e:
        if dataset not in LONGBENCH_E_DATASETS:
            raise ValueError(f"{dataset} is not available in LongBench-E.")
        return f"{dataset}_e"
    return dataset


def load_longbench_records(args):
    hf_name = resolve_dataset_name(args.dataset, args.longbench_e)
    try:
        data = load_dataset(
            args.hf_repo,
            hf_name,
            split=args.split,
            trust_remote_code=True,
        )
        records = [dict(item) for item in data]
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        records = load_longbench_records_from_zip(args.hf_repo, hf_name)
    for index, record in enumerate(records):
        record.setdefault("dataset", args.dataset)
        record.setdefault("_id", record.get("id", index))
    return records


def load_longbench_records_from_zip(hf_repo, dataset_name):
    zip_path = hf_hub_download(
        repo_id=hf_repo,
        repo_type="dataset",
        filename="data.zip",
    )
    member = f"data/{dataset_name}.jsonl"
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if line:
                    records.append(json.loads(line))
    return records


def longbench_prompt(context, question, record):
    dataset = str(record.get("dataset", "")).lower()
    if dataset.endswith("_e"):
        dataset = dataset[:-2]
    template = DATASET2PROMPT.get(dataset)
    if template is None:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    return template.format(
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
    args = parse_args()
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
