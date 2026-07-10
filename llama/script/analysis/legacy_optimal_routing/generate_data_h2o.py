"""
Generate compact H2O routing training data with per-head oracle bias Mj.

For each sample from wikitext train split:
1) Build online baseline H2O patches for prefix layers (0..L-1).
2) On target layer L and target head h, optimize per-key bias Mj with v_l2_gt objective.
3) Export one JSON per (sample, L, h) under:
   data_h2o/budget_b/layer_L/head_h/samplex.json

Each JSON stores:
- one shared qkv block (all q/k/v vectors for this layer/head)
- per_q entries keyed by q index, each containing only lightweight kept items:
    kept index, heavy hitter accumulated score, is_hh, and oracle Mj.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..attention import build_qk_routing_alpha, gen_mask, get_attention_map_after_rope
from .compare_q_bias import optimize_bias_v_l2
from ..context import RunContext
from ..online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ..sanity import build_modified_attn_hidden, move_model_inputs_to_device


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_torch_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate H2O training data with per-head optimal Mj, "
            "saved as per-sample JSON files."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--start-sample", type=int, default=0)

    parser.add_argument("--layers", type=int, nargs="+", required=True)

    parser.add_argument("--strategy", type=str, default="h2o", choices=["h2o"])
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--adaptive-budget", action="store_true")

    parser.add_argument("--bias-steps", type=int, default=500)
    parser.add_argument("--bias-lr", type=float, default=5e-2)
    parser.add_argument("--bias-l2", type=float, default=0.0)

    parser.add_argument("--output-root", type=str, default="data_h2o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def _build_wikitext_train_prompt():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return "\n".join([text for text in dataset["text"] if text.strip()])

def _prepare_token_stream(tokenizer, dataset_name):
    if dataset_name != "wikitext":
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Only wikitext is supported.")

    prompt = _build_wikitext_train_prompt()
    encoded = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise RuntimeError("Tokenizer output does not contain input_ids.")
    return encoded


def _validate_requested_spans(encoded, seq_len, start_sample, sample_count):
    if seq_len <= 0:
        raise ValueError("--seq-len must be > 0")
    if start_sample < 0:
        raise ValueError("--start-sample must be >= 0")
    if sample_count <= 0:
        raise ValueError("--sample-count must be > 0")

    total_len = int(encoded["input_ids"].shape[1])
    required_len = (start_sample + sample_count) * seq_len + 1
    if total_len < required_len:
        raise ValueError(
            f"Tokenized length ({total_len}) is shorter than required ({required_len}). "
            "Try reducing sample count/start or seq-len."
        )


def _make_sample_context(base_ctx, encoded, sample_id, seq_len):
    token_start = int(sample_id * seq_len)
    token_end = token_start + seq_len

    inputs = {k: v[:, token_start:token_end] for k, v in encoded.items()}
    gt_label = encoded["input_ids"][:, token_start + 1 : token_end + 1]

    return (
        RunContext(
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
        ),
        token_start,
    )


def _resolve_and_validate_indices(args, model_config):
    n_layers = int(model_config.num_hidden_layers)
    n_heads = int(model_config.num_attention_heads)

    layers = sorted(set(int(x) for x in args.layers))
    heads = list(range(n_heads))

    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Invalid --layers entry {layer}; expected [0, {n_layers - 1}]")

    if args.budget <= 0 or args.budget > 1:
        raise ValueError("--budget must be in (0, 1]")

    return layers, heads


def _visible_tokens(seq_len, budget, adaptive_budget, layer_idx):
    visible = int(seq_len * budget)
    if adaptive_budget and (layer_idx == 0 or layer_idx == 1):
        return seq_len
    return max(1, visible)


def _build_prefix_layer_contexts(ctx, args, layers, pos_list, model_inputs):
    """Build runtime layer contexts under baseline H2O prefix patches."""
    max_layer = max(layers)
    all_heads = list(range(int(ctx.model_config.num_attention_heads)))

    layer_ctx_by_layer = {}
    layer_to_patch = {}

    for layer_idx in range(max_layer + 1):
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        layer_ctx_by_layer[layer_idx] = layer_ctx

        if layer_idx < max_layer:
            mask_all = gen_mask(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                head_idx=all_heads,
                strategy=args.strategy,
                budget=args.budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
            )
            alpha_all = build_qk_routing_alpha(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                head_idx=all_heads,
                pos_list=pos_list,
                mask=mask_all,
                device=ctx.device,
            )
            layer_to_patch[layer_idx] = build_modified_attn_hidden(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                head_idx=all_heads,
                pos_list=pos_list,
                alpha=alpha_all,
                device=ctx.device,
            )

    return layer_ctx_by_layer


def _simulate_h2o_cache_trace(attn_head, seq_len, visible):
    """Simulate H2O cache dynamics for one head, returning per-q selected keys and stats."""
    recent_budget = visible // 2
    trace = []

    acc = torch.zeros(seq_len, device=attn_head.device, dtype=torch.float32)
    in_cache = torch.zeros(seq_len, device=attn_head.device, dtype=torch.bool)

    for q_idx in range(seq_len):
        total_available = q_idx + 1

        acc[:total_available] += attn_head[q_idx, :total_available].to(torch.float32)
        in_cache[q_idx] = True

        cur_cache_size = int(in_cache[:total_available].sum().item())
        if cur_cache_size > visible:
            cache_idx = torch.nonzero(in_cache[:total_available], as_tuple=False).squeeze(-1)
            recent_start = max(0, total_available - recent_budget)
            hh_idx = cache_idx[cache_idx < recent_start]
            if hh_idx.numel() == 0:
                hh_idx = cache_idx
            victim = hh_idx[torch.argmin(acc[hh_idx])]
            in_cache[int(victim.item())] = False

        selected_idx = torch.nonzero(in_cache[:total_available], as_tuple=False).squeeze(-1)
        recent_start = max(0, total_available - recent_budget)
        is_hh = selected_idx < recent_start
        selected_scores = acc[selected_idx]

        trace.append(
            {
                "selected_idx": selected_idx.detach().cpu(),
                "scores": selected_scores.detach().cpu(),
                "is_hh": is_hh.detach().cpu(),
            }
        )

    return trace


def _build_single_json_payload(layer_ctx, layer_idx, head_idx, seq_len, budget, adaptive_budget, token_start, args):
    head_list = [int(head_idx)]
    pos_list = list(range(seq_len))

    mask = gen_mask(
        ctx=layer_ctx,
        layer_idx=layer_idx,
        pos_list=pos_list,
        head_idx=head_list,
        strategy=args.strategy,
        budget=budget,
        seq_len=seq_len,
        adaptive_budget=adaptive_budget,
    )

    qk_scores, attn_soft = get_attention_map_after_rope(
        layer_ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=layer_ctx.device,
    )
    qk_logits = qk_scores[head_list][:, pos_list, :].to(torch.float32)

    v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(layer_ctx.device)[0][head_list].float()
    v_gt = (
        layer_ctx.attn_output[layer_idx]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(layer_ctx.device)[head_list]
        .float()
    )

    bias, _alpha_bias, _history = optimize_bias_v_l2(
        v_head=v_head,
        v_gt=v_gt,
        qk_logits=qk_logits,
        mask=mask,
        bias_steps=args.bias_steps,
        bias_lr=args.bias_lr,
        bias_l2=args.bias_l2,
    )
    bias_vec = bias[0].detach().cpu().to(torch.float32)

    visible = _visible_tokens(
        seq_len=seq_len,
        budget=budget,
        adaptive_budget=adaptive_budget,
        layer_idx=layer_idx,
    )
    trace = _simulate_h2o_cache_trace(
        attn_head=attn_soft[int(head_idx)],
        seq_len=seq_len,
        visible=visible,
    )

    q_tensor = layer_ctx.rope_qkv[layer_idx]["q"][0, int(head_idx)].detach().cpu().to(torch.float32)
    k_tensor = layer_ctx.rope_qkv[layer_idx]["k"][0, int(head_idx)].detach().cpu().to(torch.float32)
    v_tensor = layer_ctx.rope_qkv[layer_idx]["v"][0, int(head_idx)].detach().cpu().to(torch.float32)

    per_q = {}
    for q_idx in range(seq_len):
        q_key = str(q_idx)

        kv_entries = []
        selected_idx = trace[q_idx]["selected_idx"]
        selected_scores = trace[q_idx]["scores"]
        selected_is_hh = trace[q_idx]["is_hh"]

        # Keep a strict consistency check against mask-derived visible indices.
        finite_idx = torch.nonzero(
            torch.isfinite(mask[0, q_idx, : q_idx + 1]),
            as_tuple=False,
        ).squeeze(-1).detach().cpu()
        if selected_idx.numel() != finite_idx.numel() or not torch.equal(selected_idx, finite_idx):
            raise RuntimeError(
                f"H2O trace mismatch with mask at q={q_idx}, layer={layer_idx}, head={head_idx}."
            )

        for t in range(int(selected_idx.numel())):
            j_local = int(selected_idx[t].item())
            kv_entries.append(
                {
                    "kept_index": int(token_start + j_local),
                    "kept_index_local": j_local,
                    "heavyhitter_score": float(selected_scores[t].item()),
                    "is_hh": bool(selected_is_hh[t].item()),
                    "oracle_mj": float(bias_vec[j_local].item()),
                }
            )

        per_q[q_key] = {
            "q_index": int(q_idx),
            "q_global_index": int(token_start + q_idx),
            "kept": kv_entries,
        }

    payload = {
        "meta": {
            "layer": int(layer_idx),
            "head": int(head_idx),
            "seq_len": int(seq_len),
            "budget": float(budget),
            "token_start": int(token_start),
        },
        "qkv": {
            "q": q_tensor.tolist(),
            "k": k_tensor.tolist(),
            "v": v_tensor.tolist(),
        },
        "per_q": per_q,
    }
    return payload


def _sample_output_path(output_root, budget, layer_idx, head_idx, sample_id):
    budget_tag = f"budget_{float(budget):g}"
    return os.path.join(
        output_root,
        budget_tag,
        f"layer_{int(layer_idx)}",
        f"head_{int(head_idx)}",
        f"sample{int(sample_id)}.json",
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    dtype = str_to_torch_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.eval()

    base_ctx = RunContext(
        model=model,
        tokenizer=tokenizer,
        rope_qkv=None,
        inputs=None,
        outputs=None,
        attn_output=None,
        layer_input=None,
        gt_label=None,
        model_config=model.config,
        dtype=dtype,
        device=args.device,
    )

    layers, heads = _resolve_and_validate_indices(args, model.config)

    print("Loading train token stream...")
    encoded = _prepare_token_stream(tokenizer, args.dataset)
    _validate_requested_spans(
        encoded=encoded,
        seq_len=args.seq_len,
        start_sample=args.start_sample,
        sample_count=args.sample_count,
    )

    print(
        f"Start generation: samples={args.sample_count}, seq_len={args.seq_len}, "
        f"layers={layers}, heads={heads}, budget={args.budget:g}"
    )

    pos_list = list(range(args.seq_len))

    for offset in range(args.sample_count):
        sample_id = int(args.start_sample + offset)

        sample_ctx, token_start = _make_sample_context(
            base_ctx=base_ctx,
            encoded=encoded,
            sample_id=sample_id,
            seq_len=args.seq_len,
        )
        model_inputs = move_model_inputs_to_device(sample_ctx.inputs, sample_ctx.device)

        print(f"\n[sample {sample_id}] build prefix layer contexts")
        layer_ctx_by_layer = _build_prefix_layer_contexts(
            ctx=sample_ctx,
            args=args,
            layers=layers,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

        for layer_idx in layers:
            layer_ctx = layer_ctx_by_layer[layer_idx]
            for head_idx in heads:
                print(f"[sample {sample_id}] layer={layer_idx} head={head_idx} export json")
                payload = _build_single_json_payload(
                    layer_ctx=layer_ctx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    seq_len=args.seq_len,
                    budget=args.budget,
                    adaptive_budget=args.adaptive_budget,
                    token_start=token_start,
                    args=args,
                )

                out_path = _sample_output_path(
                    output_root=args.output_root,
                    budget=args.budget,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    sample_id=sample_id,
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=args.indent, ensure_ascii=False)

                print(f"Saved: {out_path}")

    print("All done.")


if __name__ == "__main__":
    main()
