import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...attention import build_qk_routing_alpha, gen_mask
from ...context import RunContext
from ...online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ...runner_utils import build_prompt, mean_nll_and_ppl, set_seed, str_to_torch_dtype
from ...sanity import (
    build_modified_attn_hidden,
    compute_metrics,
    grouped_query_heads,
    get_tail_labels,
    move_model_inputs_to_device,
)


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parents[4] / "result"
LEGACY_PROTOCOL = "legacy"
ALIGNED_PROTOCOL = "aligned"


def model_output_name(model):
    return str(model).rstrip("/").split("/")[-1]


DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / model_output_name(DEFAULT_MODEL) / "wikitext_n100"


def add_common_args(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    parser.add_argument(
        "--ppl-protocol",
        choices=[LEGACY_PROTOCOL, ALIGNED_PROTOCOL],
        default=LEGACY_PROTOCOL,
        help=(
            "legacy preserves the original cross-window target convention; "
            "aligned uses standard within-window next-token labels and the "
            "WikiText-2 double-newline concatenation used by compression work."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
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
    if args.output_root is None:
        result_name = (
            "aligned_ppl"
            if args.ppl_protocol == ALIGNED_PROTOCOL
            else "wikitext_n100"
        )
        args.output_root = (
            DEFAULT_RESULT_ROOT / model_output_name(args.model) / result_name
        )
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
    target_lookahead = 0 if args.ppl_protocol == ALIGNED_PROTOCOL else 1
    required_len = starts[-1] + args.seq_len + target_lookahead
    join_separator = "\n\n" if args.ppl_protocol == ALIGNED_PROTOCOL else "\n"
    prompt = build_prompt(
        args.dataset,
        join_separator=join_separator,
        filter_empty=args.ppl_protocol != ALIGNED_PROTOCOL,
    )
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


def build_sample_context(
    model,
    tokenizer,
    encoded,
    dtype,
    device,
    start,
    seq_len,
    ppl_protocol=LEGACY_PROTOCOL,
):
    inputs = {
        key: value[:, start : start + seq_len]
        for key, value in encoded.items()
    }
    if ppl_protocol == ALIGNED_PROTOCOL:
        # Match the standard causal-LM shift inside an independent fixed-length
        # evaluation chunk. No label is read from the following chunk.
        labels = encoded["input_ids"][:, start + 1 : start + seq_len]
    else:
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
    pos_list = list(range(ctx.gt_label.shape[1]))
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
        pos_list=pos_list,
    )


def run_attention_topk_method(
    *,
    ctx,
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

    patches = {}
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
        alpha = build_attention_topk_alpha_from_artifacts(
            artifacts=artifacts,
            pos_list=pos_list,
            budget=budget,
            seq_len=seq_len,
            device=ctx.device,
            model_config=ctx.model_config,
        )
        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=list(range(ctx.model_config.num_attention_heads)),
            pos_list=pos_list,
            alpha=alpha,
            device=ctx.device,
        )
        del artifacts, layer_ctx, alpha
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
        pos_list=pos_list,
    )


def build_attention_topk_alpha_from_artifacts(
    *,
    artifacts,
    pos_list,
    budget,
    seq_len,
    device,
    model_config=None,
):
    q_all = artifacts["q"].to(device)[0].float()
    k_all = artifacts["k"].to(device)[0].float()
    pos_tensor = torch.tensor(pos_list, device=device, dtype=torch.long)
    q_pos = q_all[:, pos_tensor, :]
    scale = math.sqrt(q_all.shape[-1])
    qk_logits = torch.empty(
        q_all.shape[0],
        len(pos_list),
        seq_len,
        device=device,
        dtype=torch.float32,
    )
    groups = grouped_query_heads(
        list(range(q_all.shape[0])),
        model_config,
        num_kv_heads=k_all.shape[0],
    )
    for kv_head, out_indices, query_heads in groups:
        qk_logits[out_indices] = (
            torch.einsum("hqd,kd->hqk", q_pos[query_heads], k_all[kv_head]) / scale
        )

    key_idx = torch.arange(seq_len, device=device)
    causal = key_idx.view(1, 1, seq_len) <= pos_tensor.view(1, -1, 1)
    visible = max(1, int(seq_len * budget))

    if visible >= seq_len:
        selected = causal.expand(qk_logits.shape[0], -1, -1)
    else:
        selected = torch.zeros_like(qk_logits, dtype=torch.bool)
        for _kv_head, out_indices, _query_heads in groups:
            group_logits = qk_logits[out_indices]
            select_logits = group_logits.mean(dim=0).masked_fill(
                ~causal[0],
                float("-inf"),
            )
            topk_idx = torch.topk(
                select_logits,
                k=visible,
                dim=-1,
                largest=True,
            ).indices
            group_selected = torch.zeros_like(select_logits, dtype=torch.bool)
            group_selected.scatter_(dim=-1, index=topk_idx, value=True)
            for out_idx in out_indices:
                selected[out_idx] = group_selected
        selected &= causal

    alpha_logits = qk_logits.masked_fill(~selected, float("-inf"))
    return F.softmax(alpha_logits, dim=-1)


def effective_causal_budget(
    seq_len,
    budget,
    num_layers,
    full_attention_layers,
    pos_list=None,
):
    visible = int(seq_len * budget)
    query_positions = range(seq_len) if pos_list is None else pos_list
    total_available = sum(pos + 1 for pos in query_positions)
    selected_per_compressed_layer = sum(
        min(pos + 1, visible) for pos in query_positions
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
        "num_tokens": int(labels.numel()),
    }


def load_or_create_summary(path, method, args, settings):
    if path.exists():
        summary = torch.load(path, map_location="cpu", weights_only=False)
        expected_starts = sample_starts(args)
        if summary.get("starts") != expected_starts:
            raise ValueError(
                f"Existing starts in {path} do not match current configuration."
            )
        existing_config = summary.get("config", {})
        existing_protocol = existing_config.get("ppl_protocol", LEGACY_PROTOCOL)
        if existing_protocol != args.ppl_protocol:
            raise ValueError(
                f"Existing protocol {existing_protocol!r} in {path} does not "
                f"match requested protocol {args.ppl_protocol!r}."
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
    teacher_token_counts = []
    for sample in summary["samples"].values():
        teacher_ppls.append(float(sample["teacher_ppl"]))
        teacher_nlls.append(float(sample["teacher_nll"]))
        teacher_token_counts.append(int(sample.get("num_tokens", 1)))
        for setting, record in sample["results"].items():
            grouped.setdefault(float(setting), []).append(record)

    aggregate = {
        "num_completed_samples": len(summary["samples"]),
    }
    if teacher_ppls:
        aggregate["teacher"] = summarize_values(
            teacher_ppls,
            teacher_nlls,
            teacher_token_counts,
        )

    aggregate["settings"] = {}
    for setting, records in sorted(grouped.items()):
        ppls = [float(record["student_ppl"]) for record in records]
        nlls = [float(record["student_nll"]) for record in records]
        budgets = [float(record["measured_budget"]) for record in records]
        token_counts = [int(record.get("num_tokens", 1)) for record in records]
        setting_summary = summarize_values(ppls, nlls, token_counts)
        setting_summary["mean_measured_budget"] = float(
            torch.tensor(budgets, dtype=torch.float64).mean().item()
        )
        aggregate["settings"][setting] = setting_summary
    return aggregate


def summarize_values(ppls, nlls, token_counts=None):
    ppl_tensor = torch.tensor(ppls, dtype=torch.float64)
    nll_tensor = torch.tensor(nlls, dtype=torch.float64)
    if token_counts is None:
        token_count_tensor = torch.ones_like(nll_tensor)
    else:
        token_count_tensor = torch.tensor(token_counts, dtype=torch.float64)
    corpus_nll = (nll_tensor * token_count_tensor).sum() / token_count_tensor.sum()
    return {
        "count": int(ppl_tensor.numel()),
        "mean_ppl": float(ppl_tensor.mean().item()),
        "std_ppl": float(ppl_tensor.std(unbiased=False).item()),
        "mean_nll": float(nll_tensor.mean().item()),
        "num_tokens": int(token_count_tensor.sum().item()),
        "corpus_nll": float(corpus_nll.item()),
        "corpus_ppl": float(torch.exp(corpus_nll).item()),
    }


def plot_aggregate(summary, path, setting_label):
    settings = summary["aggregate"].get("settings", {})
    if not settings:
        return
    protocol = summary.get("config", {}).get("ppl_protocol", LEGACY_PROTOCOL)
    ppl_key = "corpus_ppl" if protocol == ALIGNED_PROTOCOL else "mean_ppl"
    points = sorted(
        (
            float(record["mean_measured_budget"]),
            float(record[ppl_key]),
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
        teacher_label = (
            "Full-attention corpus PPL"
            if protocol == ALIGNED_PROTOCOL
            else "Mean full-attention PPL"
        )
        ax.axhline(
            float(teacher[ppl_key]),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            label=teacher_label,
        )
    ax.set_xlabel("Mean equivalent causal attention budget")
    ax.set_ylabel("Corpus PPL" if protocol == ALIGNED_PROTOCOL else "Mean sample PPL")
    if protocol == ALIGNED_PROTOCOL:
        ax.set_xscale("log")
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
    plot_name = (
        "corpus_ppl_vs_budget.png"
        if args.ppl_protocol == ALIGNED_PROTOCOL
        else "mean_ppl_vs_budget.png"
    )
    plot_path = output_dir / plot_name
    summary = load_or_create_summary(summary_path, method, args, settings)

    model, tokenizer, encoded, dtype, starts = load_model_and_tokens(args)
    sample_iter = tqdm(
        list(enumerate(starts)),
        total=len(starts),
        desc=f"{method} samples",
        unit="sample",
        dynamic_ncols=True,
    )
    for sample_idx, start in sample_iter:
        sample_key = int(sample_idx)
        if sample_key in summary["samples"]:
            sample_iter.set_postfix(sample=sample_idx, start=start, status="skip")
            continue
        sample_iter.set_postfix(sample=sample_idx, start=start, status="run")
        ctx = build_sample_context(
            model=model,
            tokenizer=tokenizer,
            encoded=encoded,
            dtype=dtype,
            device=args.device,
            start=start,
            seq_len=args.seq_len,
            ppl_protocol=args.ppl_protocol,
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
            "num_tokens": int(labels.numel()),
            "results": results,
        }
        save_summary(summary, summary_path)
        plot_aggregate(summary, plot_path, setting_label)
        del ctx, ref_logits, results
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {plot_path}")
