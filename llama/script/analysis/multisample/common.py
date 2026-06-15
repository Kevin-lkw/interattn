import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..attention import build_qk_routing_alpha, gen_mask
from ..context import RunContext
from ..online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ..runner_utils import build_prompt, mean_nll_and_ppl, set_seed, str_to_torch_dtype
from ..sanity import (
    build_modified_attn_hidden,
    compute_metrics,
    get_tail_labels,
    move_model_inputs_to_device,
)


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3] / "result" / "wikitext_n100"
)


def add_common_args(parser):
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Token distance between windows. Defaults to seq_len.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def validate_common_args(args):
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.start_offset < 0:
        raise ValueError("--start-offset must be >= 0")
    if args.sample_stride is None:
        args.sample_stride = args.seq_len
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be > 0")
    return args


def sample_starts(args):
    return [
        args.start_offset + sample_idx * args.sample_stride
        for sample_idx in range(args.num_samples)
    ]


def load_model_and_tokens(args):
    dtype = str_to_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    starts = sample_starts(args)
    required_len = starts[-1] + args.seq_len + 1
    prompt = build_prompt(args.dataset)
    encoded = tokenizer(
        prompt,
        max_length=required_len,
        truncation=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] < required_len:
        raise ValueError(
            f"Token stream has {encoded['input_ids'].shape[1]} tokens, "
            f"but {required_len} are required."
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer, encoded, dtype, starts


def build_sample_context(model, tokenizer, encoded, dtype, device, start, seq_len):
    inputs = {
        key: value[:, start : start + seq_len]
        for key, value in encoded.items()
    }
    labels = encoded["input_ids"][:, start + 1 : start + seq_len + 1]
    return RunContext(
        model=model,
        tokenizer=tokenizer,
        rope_qkv=None,
        inputs=inputs,
        outputs=None,
        attn_output=None,
        layer_input=None,
        gt_label=labels,
        model_config=model.config,
        dtype=dtype,
        device=device,
    )


def prepare_sample(ctx, seq_len):
    pos_list = list(range(seq_len))
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)
    with torch.no_grad():
        ref_logits = ctx.model(
            **model_inputs, use_cache=False
        ).logits[:, pos_list, :].float()
    teacher_nll, teacher_ppl = mean_nll_and_ppl(ref_logits, labels)
    return pos_list, model_inputs, labels, ref_logits, teacher_nll, teacher_ppl


def run_routing_method(
    *,
    ctx,
    strategy,
    budget,
    full_attention_layers,
    seq_len,
    pos_list,
    model_inputs,
):
    if math.isclose(float(budget), 1.0):
        with torch.no_grad():
            logits = ctx.model(
                **model_inputs, use_cache=False
            ).logits[:, pos_list, :].float()
        return logits, 1.0

    head_idx = list(range(ctx.model_config.num_attention_heads))
    patches = {}
    mask_args = SimpleNamespace(
        adaptive_budget=False,
    )
    for layer_idx in range(ctx.model_config.num_hidden_layers):
        if layer_idx < full_attention_layers:
            continue
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=strategy,
            budget=budget,
            seq_len=seq_len,
            adaptive_budget=mask_args.adaptive_budget,
        )
        alpha = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=mask,
            device=ctx.device,
        )
        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha,
            device=ctx.device,
        )
        del artifacts, layer_ctx, mask, alpha
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    logits = run_with_multilayer_patches(
        ctx=ctx,
        layer_to_patch=patches,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )
    return logits, effective_causal_budget(
        seq_len=seq_len,
        budget=budget,
        num_layers=ctx.model_config.num_hidden_layers,
        full_attention_layers=full_attention_layers,
    )


def effective_causal_budget(seq_len, budget, num_layers, full_attention_layers):
    visible = int(seq_len * budget)
    total_available = seq_len * (seq_len + 1) // 2
    selected_per_compressed_layer = sum(
        min(pos + 1, visible) for pos in range(seq_len)
    )
    selected = (
        full_attention_layers * total_available
        + (num_layers - full_attention_layers) * selected_per_compressed_layer
    )
    return selected / (num_layers * total_available)


def metric_record(ref_logits, student_logits, labels, measured_budget):
    metrics = compute_metrics(ref_logits, student_logits, labels)
    return {
        **metrics,
        "student_ppl": math.exp(float(metrics["student_nll"])),
        "teacher_ppl": math.exp(float(metrics["teacher_nll"])),
        "measured_budget": float(measured_budget),
    }


def load_or_create_summary(path, method, args, settings):
    if path.exists():
        summary = torch.load(path, map_location="cpu", weights_only=False)
        expected_starts = sample_starts(args)
        if summary.get("starts") != expected_starts:
            raise ValueError(
                f"Existing starts in {path} do not match current configuration."
            )
        return summary
    return {
        "method": method,
        "config": vars(args),
        "starts": sample_starts(args),
        "settings": [float(value) for value in settings],
        "samples": {},
        "aggregate": {},
    }


def save_summary(summary, path):
    summary["aggregate"] = aggregate_samples(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(summary, tmp_path)
    os.replace(tmp_path, path)


def aggregate_samples(summary):
    grouped = {}
    teacher_ppls = []
    teacher_nlls = []
    for sample in summary["samples"].values():
        teacher_ppls.append(float(sample["teacher_ppl"]))
        teacher_nlls.append(float(sample["teacher_nll"]))
        for setting, record in sample["results"].items():
            grouped.setdefault(float(setting), []).append(record)

    aggregate = {
        "num_completed_samples": len(summary["samples"]),
    }
    if teacher_ppls:
        aggregate["teacher"] = summarize_values(teacher_ppls, teacher_nlls)

    aggregate["settings"] = {}
    for setting, records in sorted(grouped.items()):
        ppls = [float(record["student_ppl"]) for record in records]
        nlls = [float(record["student_nll"]) for record in records]
        budgets = [float(record["measured_budget"]) for record in records]
        setting_summary = summarize_values(ppls, nlls)
        setting_summary["mean_measured_budget"] = float(
            torch.tensor(budgets, dtype=torch.float64).mean().item()
        )
        aggregate["settings"][setting] = setting_summary
    return aggregate


def summarize_values(ppls, nlls):
    ppl_tensor = torch.tensor(ppls, dtype=torch.float64)
    nll_tensor = torch.tensor(nlls, dtype=torch.float64)
    return {
        "count": int(ppl_tensor.numel()),
        "mean_ppl": float(ppl_tensor.mean().item()),
        "std_ppl": float(ppl_tensor.std(unbiased=False).item()),
        "mean_nll": float(nll_tensor.mean().item()),
        "corpus_ppl": float(torch.exp(nll_tensor.mean()).item()),
    }


def plot_aggregate(summary, path, setting_label):
    settings = summary["aggregate"].get("settings", {})
    if not settings:
        return
    points = sorted(
        (
            float(record["mean_measured_budget"]),
            float(record["mean_ppl"]),
            float(setting),
        )
        for setting, record in settings.items()
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        marker="o",
        linewidth=1.6,
    )
    for budget, ppl, setting in points:
        ax.annotate(
            f"{setting_label}={setting:g}",
            (budget, ppl),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    teacher = summary["aggregate"].get("teacher")
    if teacher:
        ax.axhline(
            float(teacher["mean_ppl"]),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            label="Mean full-attention PPL",
        )
    ax.set_xlabel("Mean equivalent causal attention budget")
    ax.set_ylabel("Mean sample PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.24)
    ax.legend()
    ax.set_title(
        f"{summary['method']}: {summary['aggregate']['num_completed_samples']} samples"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_multisample(args, method, settings, evaluate_sample, setting_label):
    validate_common_args(args)
    set_seed(args.seed)
    output_dir = args.output_root / method
    summary_path = output_dir / "summary.pt"
    plot_path = output_dir / "mean_ppl_vs_budget.png"
    summary = load_or_create_summary(summary_path, method, args, settings)

    model, tokenizer, encoded, dtype, starts = load_model_and_tokens(args)
    for sample_idx, start in enumerate(starts):
        sample_key = int(sample_idx)
        if sample_key in summary["samples"]:
            print(f"[skip] sample={sample_idx} start={start}")
            continue
        ctx = build_sample_context(
            model=model,
            tokenizer=tokenizer,
            encoded=encoded,
            dtype=dtype,
            device=args.device,
            start=start,
            seq_len=args.seq_len,
        )
        prepared = prepare_sample(ctx, args.seq_len)
        pos_list, model_inputs, labels, ref_logits, teacher_nll, teacher_ppl = prepared
        print(
            f"[sample {sample_idx + 1}/{len(starts)}] start={start}, "
            f"teacher_ppl={teacher_ppl:.6f}"
        )
        results = evaluate_sample(
            ctx=ctx,
            args=args,
            settings=settings,
            pos_list=pos_list,
            model_inputs=model_inputs,
            labels=labels,
            ref_logits=ref_logits,
        )
        summary["samples"][sample_key] = {
            "start": int(start),
            "teacher_nll": float(teacher_nll),
            "teacher_ppl": float(teacher_ppl),
            "results": results,
        }
        save_summary(summary, summary_path)
        plot_aggregate(summary, plot_path, setting_label)
        del ctx, ref_logits, results
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {plot_path}")
