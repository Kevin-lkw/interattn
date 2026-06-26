import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..runner_utils import set_seed, str_to_torch_dtype
from .methods import add_method_args, build_method, generate_with_method


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "result" / "generate"
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
LOCAL_PATCH_METHODS = {"attention_topk", "condition_block", "quest"}


def add_generation_args(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="JSON/JSONL input file. Benchmark-specific runners may load data directly.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Do not wrap prompts with the tokenizer chat template.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answers")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--id-field", default="id")
    add_method_args(parser)
    return parser


def validate_generation_args(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.method == "condition_block":
        if args.budget is not None:
            raise ValueError("--budget is not used by condition_block; use --condition-block-size and --condition-eps.")
        if args.condition_block_size is None or args.condition_block_size <= 0:
            raise ValueError("--condition-block-size must be provided and > 0 for condition_block.")
    elif args.method == "quest":
        has_budget = args.budget is not None
        has_page_size = args.quest_page_size is not None
        if has_budget == has_page_size:
            raise ValueError("quest requires exactly one of --budget or --quest-page-size.")
        if has_budget and not 0 < float(args.budget) <= 1:
            raise ValueError("--budget must be in (0, 1]")
        if has_page_size and args.quest_page_size <= 0:
            raise ValueError("--quest-page-size must be > 0")
    else:
        if args.budget is not None and not 0 < float(args.budget) <= 1:
            raise ValueError("--budget must be in (0, 1]")
    return args


def load_model_and_tokenizer(args):
    dtype = str_to_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "dtype": dtype,
        "device_map": {"": args.device},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.method == "h2o" and args.attn_implementation is None:
        args.attn_implementation = "eager"
    if args.method in LOCAL_PATCH_METHODS:
        if args.attn_implementation is None:
            args.attn_implementation = "eager"
        elif args.attn_implementation != "eager":
            raise ValueError(
                f"--method {args.method} requires --attn-implementation eager "
                "because it captures Q/K/V through the eager attention path."
            )
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, tokenizer


def _dataset_name(record):
    dataset = str(record.get("dataset", "")).lower()
    if dataset.endswith("_e"):
        dataset = dataset[:-2]
    return dataset


def _middle_truncate_prompt(prompt, tokenizer, max_input_tokens):
    if max_input_tokens is None:
        return prompt
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_input_tokens:
        return prompt
    half = max_input_tokens // 2
    return tokenizer.decode(
        tokenized[:half],
        skip_special_tokens=True,
    ) + tokenizer.decode(
        tokenized[-half:],
        skip_special_tokens=True,
    )


def _maybe_apply_chat_template(prompt, tokenizer, args, record):
    if args.disable_chat_template:
        return prompt
    if _dataset_name(record) in NO_CHAT_DATASETS:
        return prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(
        "Chat template is enabled, but the tokenizer does not provide chat_template. "
        "Pass --disable-chat-template if you intentionally want raw prompts."
    )


def read_records(path, limit=None):
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "data" in payload:
            payload = payload["data"]
        elif "examples" in payload:
            payload = payload["examples"]
        else:
            payload = list(payload.values())
    if limit is not None:
        payload = payload[:limit]
    return payload


def build_prompt(record, args, prompt_builder):
    if args.prompt_field and record.get(args.prompt_field) is not None:
        return str(record[args.prompt_field])
    context = str(record.get(args.context_field, ""))
    question = str(record.get(args.question_field, ""))
    return prompt_builder(context, question, record)


def default_prompt_builder(context, question, _record):
    if question:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    return context


def extract_answers(record, args):
    value = record.get(args.answer_field, None)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def record_id(record, args, index):
    return str(record.get(args.id_field, index))


def output_path(args, benchmark_name):
    model_name = str(args.model).rstrip("/").split("/")[-1]
    method = build_method(args)
    if method.kind == "condition_block":
        filename = (
            f"{method.name}_block={method.condition_block_size}"
            f"_eps={method.condition_eps:g}.jsonl"
        )
    elif method.kind == "quest" and args.budget is None:
        filename = f"{method.name}_page={method.quest_page_size}.jsonl"
    else:
        filename = f"{method.name}_budget={method.budget:g}.jsonl"
    out_dir = args.output_root / model_name / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def pending_generation_records(args, benchmark_name, records=None):
    if records is None:
        if args.data is None:
            raise ValueError("--data is required unless the benchmark runner provides records.")
        records = read_records(args.data, args.limit)
    elif args.limit is not None:
        records = records[: args.limit]
    out_path = output_path(args, benchmark_name)
    done_ids = _load_done_ids(out_path)
    pending_records = [
        (index, record)
        for index, record in enumerate(records)
        if record_id(record, args, index) not in done_ids
    ]
    return records, out_path, done_ids, pending_records


def run_generation_benchmark(
    args,
    benchmark_name,
    prompt_builder=default_prompt_builder,
    records=None,
    model=None,
    tokenizer=None,
):
    validate_generation_args(args)
    if args.max_new_tokens is None:
        args.max_new_tokens = 32
    set_seed(args.seed)
    method = build_method(args)
    records, out_path, done_ids, pending_records = pending_generation_records(
        args,
        benchmark_name,
        records,
    )
    if not pending_records:
        print(f"All predictions already exist; skipping model load: {out_path}")
        return out_path
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(args)
    if args.max_input_tokens is None and getattr(model.config, "max_position_embeddings", None):
        args.max_input_tokens = int(model.config.max_position_embeddings)

    with out_path.open("a", encoding="utf-8") as handle:
        for index, record in tqdm(
            pending_records,
            desc=f"{benchmark_name}:{method.name}",
            unit="sample",
            total=len(pending_records),
        ):
            rid = record_id(record, args, index)
            if rid in done_ids:
                continue
            prompt = build_prompt(record, args, prompt_builder)
            prompt = _middle_truncate_prompt(prompt, tokenizer, args.max_input_tokens)
            prompt = _maybe_apply_chat_template(prompt, tokenizer, args, record)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            generation_result = generate_with_method(
                model=model,
                tokenizer=tokenizer,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get(
                    "attention_mask",
                    torch.ones_like(inputs["input_ids"], device=args.device),
                ),
                method=method,
                device=args.device,
                dataset=_dataset_name(record),
            )
            if isinstance(generation_result, tuple):
                output_ids, generation_metadata = generation_result
            else:
                output_ids, generation_metadata = generation_result, {}
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            row = {
                "id": rid,
                "method": method.name,
                "pred": prediction,
                "answers": extract_answers(record, args),
                "all_classes": record.get("all_classes"),
                "length": record.get("length"),
                "input_tokens": int(inputs["input_ids"].shape[1]),
            }
            if method.budget is not None:
                row["budget"] = method.budget
            row.update(generation_metadata)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done_ids.add(rid)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"Saved predictions to: {out_path}")
    return out_path


def _load_done_ids(path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row:
                done.add(str(row["id"]))
    return done
