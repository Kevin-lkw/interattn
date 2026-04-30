import argparse
import os

import torch
from datasets import load_dataset
from torch.nn import functional as F

from .context import RunContext
from .runner import normalize_budget_key


def create_base_runner_parser(
    description: str,
    strategy_choices,
    default_strategy: str = "h2o",
    strategy_help: str = None,
    eval_start_help: str = None,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")

    strategy_kwargs = {
        "type": str,
        "default": default_strategy,
        "choices": list(strategy_choices),
    }
    if strategy_help is not None:
        strategy_kwargs["help"] = strategy_help
    parser.add_argument("--strategy", **strategy_kwargs)

    parser.add_argument("--start", type=int, default=0)
    if eval_start_help is None:
        eval_start_help = (
            "Start index for evaluation sample. If not set, use --start. "
            "Use a different value to evaluate on another sample."
        )

    parser.add_argument(
        "--eval-start",
        type=int,
        default=None,
        help=eval_start_help,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1])
    parser.add_argument("--output-dir", type=str, default=None)

    return parser


def add_tau_target_arg(
    parser,
    default: str = "v_l2_gt",
    choices=("v_l2_gt",),
    help_text: str = "Fixed target for this runner.",
):
    parser.add_argument(
        "--tau-target",
        type=str,
        default=default,
        choices=list(choices),
        help=help_text,
    )


def finalize_runner_args(args):
    if args.layers is None:
        args.all_layers = True
    else:
        args.all_layers = False
    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def resolve_output_dir(args, runner_name: str):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = adaptive_budget_tag(args.adaptive_budget)
        sample_tag = (
            f"fit{args.start}"
            if args.eval_start == args.start
            else f"fit{args.start}_eval{args.eval_start}"
        )
        out_dir = (
            f"../result/{args.dataset}_{sample_tag}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/{runner_name}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def adaptive_budget_tag(adaptive_budget):
    return "adaptive" if adaptive_budget else "fixed"


def nll_to_ppl(nll):
    return float(torch.exp(torch.tensor(float(nll), dtype=torch.float32)).item())


def load_existing_baseline_metrics(args, budget):
    metric_start = args.eval_start
    path = (
        f"../result/{args.dataset}_{metric_start}/"
        f"{adaptive_budget_tag(args.adaptive_budget)}/{args.strategy}/qk_routing.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing baseline summary file: {path}. "
            "Please run runner.py baseline comparison first."
        )

    summary = torch.load(path, map_location="cpu", weights_only=False)
    budgets = summary.get("budgets", {}) if isinstance(summary, dict) else {}
    key = normalize_budget_key(budgets, budget)
    if key is None:
        raise KeyError(f"Budget {budget} not found in baseline summary keys: {list(budgets.keys())}")

    entry = budgets[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid baseline entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {"nll": nll, "ppl": nll_to_ppl(nll), "raw": entry}


def load_existing_optimal_metrics(args, budget):
    metric_start = args.eval_start
    path = (
        f"../result/{args.dataset}_{metric_start}/"
        f"{adaptive_budget_tag(args.adaptive_budget)}/{args.strategy}/{args.loss_type}/"
        "layer_all/budget_to_final_metrics.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing optimal summary file: {path}. "
            "Please run runner.py optimization first."
        )

    summary = torch.load(path, map_location="cpu", weights_only=False)
    key = normalize_budget_key(summary, budget)
    if key is None:
        raise KeyError(f"Budget {budget} not found in optimal summary keys: {list(summary.keys())}")

    entry = summary[key]
    if "student_nll" not in entry:
        raise KeyError(f"Invalid optimal entry at budget {key}: missing student_nll")

    nll = float(entry["student_nll"])
    return {"nll": nll, "ppl": nll_to_ppl(nll), "raw": entry}


def mean_nll_and_ppl(logits, labels):
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )
    ppl = torch.exp(nll)
    return float(nll.item()), float(ppl.item())


def resolve_head_indices(num_heads):
    return list(range(num_heads))


def build_prompt(dataset_name):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join([text for text in dataset["text"] if text.strip()])
    if dataset_name == "pg19":
        dataset = load_dataset("emozilla/pg19-test", split="test")
        return "\n".join([text for text in dataset["text"] if text.strip()])
    if dataset_name == "oasst2":
        dataset = load_dataset("OpenAssistant/oasst2", split="train")
        texts = dataset["text"]
        langs = dataset["lang"] if "lang" in dataset.column_names else None
        if langs is None:
            filtered_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        else:
            filtered_texts = [
                text
                for text, lang in zip(texts, langs)
                if isinstance(text, str)
                and text.strip()
                and (lang is None or str(lang).startswith("en"))
            ]
        if len(filtered_texts) == 0:
            raise ValueError("No valid text found in OASST2 after filtering.")
        return "\n".join(filtered_texts)
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported now: wikitext, pg19, oasst2"
    )


def build_context_with_new_start(base_ctx, dataset_name, start, seq_len):
    prompt = build_prompt(dataset_name)
    encoded = base_ctx.tokenizer(
        prompt,
        max_length=start + seq_len + 1,
        truncation=True,
        return_tensors="pt",
    )

    total_len = encoded["input_ids"].shape[1]
    required_len = start + seq_len + 1
    if total_len < required_len:
        raise ValueError(
            f"Tokenized prompt length ({total_len}) is shorter than required ({required_len})."
        )

    inputs = {
        key: value[:, start : start + seq_len]
        for key, value in encoded.items()
    }
    gt_label = encoded["input_ids"][:, start + 1 : start + seq_len + 1]

    return RunContext(
        model=base_ctx.model,
        tokenizer=base_ctx.tokenizer,
        rope_qkv=None,
        inputs=inputs,
        outputs=None,
        attn_output=None,
        layer_input=None,
        gt_label=gt_label,
        model_config=base_ctx.model_config,
        dtype=base_ctx.dtype,
        device=base_ctx.device,
    )
