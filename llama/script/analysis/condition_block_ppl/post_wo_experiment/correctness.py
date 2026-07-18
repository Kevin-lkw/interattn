"""Monte Carlo correctness checks for the multi-head post-Wo derivation."""

import argparse
import json
import math
from pathlib import Path

import torch

from .core import head_spectral_norms, split_o_proj_weight


RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "post_wo_condition_block"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "correctness.json",
    )
    return parser.parse_args()


def _delta(q, keys, s_c, scale, mode):
    scores = keys @ q / scale
    exact = (scores - s_c).abs().max()
    if mode == "exact":
        return exact
    k_max = keys.max(dim=0).values
    k_min = keys.min(dim=0).values
    upper = torch.maximum(q * k_max, q * k_min).sum() / scale
    lower = torch.minimum(q * k_max, q * k_min).sum() / scale
    bound = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
    if bool(bound + 1e-12 < exact):
        raise AssertionError("range delta failed to upper-bound exact delta")
    return bound


def _head_data(q, keys, values, block_size, value_norms, delta_mode):
    scale = math.sqrt(q.numel())
    blocks = []
    for start in range(0, keys.shape[0], block_size):
        end = min(start + block_size, keys.shape[0])
        kc = keys[start:end]
        vc = values[start:end]
        k_bar = kc.mean(dim=0)
        s_c = q @ k_bar / scale
        scores = kc @ q / scale
        blocks.append(
            {
                "scores": scores,
                "values": vc,
                "v_bar": vc.mean(dim=0),
                "z_hat": math.log(end - start) + s_c,
                "delta": _delta(q, kc, s_c, scale, delta_mode),
                "b_c": value_norms[start:end].max(),
            }
        )
    z_hat = torch.stack([b["z_hat"] for b in blocks])
    p_hat = torch.softmax(z_hat, dim=0)
    delta = torch.stack([b["delta"] for b in blocks])
    b_c = torch.stack([b["b_c"] for b in blocks])
    b_all = b_c.max()
    f_value = torch.cosh(delta)
    s_f = (p_hat * f_value).sum()
    condition = p_hat * (
        2 * b_all * (f_value - 1) / s_f
        + 2 * b_c * torch.tanh(delta / 2)
    )

    all_scores = torch.cat([b["scores"] for b in blocks])
    all_values = torch.cat([b["values"] for b in blocks])
    exact = (torch.softmax(all_scores, dim=0).unsqueeze(-1) * all_values).sum(dim=0)
    approx = sum(p_hat[i] * b["v_bar"] for i, b in enumerate(blocks))
    return {
        "blocks": blocks,
        "p_hat": p_hat,
        "f_value": f_value,
        "condition": condition,
        "exact": exact,
        "approx": approx,
        "b_all": b_all,
    }


def _hybrid_output(head, selected):
    logits = []
    values = []
    for block, use_exact in zip(head["blocks"], selected):
        if use_exact:
            logits.append(block["scores"])
            values.append(block["values"])
        else:
            logits.append(block["z_hat"].reshape(1))
            values.append(block["v_bar"].reshape(1, -1))
    logits = torch.cat(logits)
    values = torch.cat(values)
    return (torch.softmax(logits, dim=0).unsqueeze(-1) * values).sum(dim=0)


def run(args):
    torch.manual_seed(args.seed)
    dtype = torch.float64
    violations = {
        "reconstruction": 0,
        "post_exact_head": 0,
        "post_exact_total": 0,
        "post_spectral_total": 0,
        "hybrid_post_exact_total": 0,
        "exact_bound_le_spectral": 0,
    }
    max_ratio = {key: 0.0 for key in violations if "reconstruction" not in key}

    for trial in range(args.trials):
        n_heads = 4
        n_kv_heads = 2
        group_size = n_heads // n_kv_heads
        head_dim = 5
        model_dim = 13
        seq_len = 5 + (trial % 13)
        block_size = 2 + (trial % 4)
        delta_mode = "exact" if trial % 2 == 0 else "range_bound"

        q = torch.randn(n_heads, head_dim, dtype=dtype)
        k = torch.randn(n_kv_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(n_kv_heads, seq_len, head_dim, dtype=dtype)
        wo = torch.randn(model_dim, n_heads * head_dim, dtype=dtype) / math.sqrt(
            n_heads * head_dim
        )
        w_heads = split_o_proj_weight(wo, n_heads, head_dim)
        spectral = head_spectral_norms(w_heads)

        pre_heads = []
        exact_post_heads = []
        spec_post_heads = []
        for h in range(n_heads):
            kv_head = h // group_size
            vh = v[kv_head]
            pre_norm = torch.linalg.vector_norm(vh, dim=-1)
            projected_norm = torch.linalg.vector_norm(
                vh @ w_heads[h].transpose(0, 1), dim=-1
            )
            pre_heads.append(
                _head_data(q[h], k[kv_head], vh, block_size, pre_norm, delta_mode)
            )
            exact_post_heads.append(
                _head_data(
                    q[h], k[kv_head], vh, block_size, projected_norm, delta_mode
                )
            )
            spec_post_heads.append(
                _head_data(
                    q[h],
                    k[kv_head],
                    vh,
                    block_size,
                    spectral[h] * pre_norm,
                    delta_mode,
                )
            )

        exact_concat = torch.cat([head["exact"] for head in pre_heads])
        projected_concat = wo @ exact_concat
        projected_sum = sum(
            w_heads[h] @ pre_heads[h]["exact"] for h in range(n_heads)
        )
        reconstruction_error = torch.linalg.vector_norm(projected_concat - projected_sum)
        if float(reconstruction_error) > 1e-10:
            violations["reconstruction"] += 1

        post_errors = []
        exact_bounds = []
        spec_bounds = []
        for h in range(n_heads):
            pre_error = pre_heads[h]["approx"] - pre_heads[h]["exact"]
            post_error = w_heads[h] @ pre_error
            post_errors.append(post_error)
            exact_bound = exact_post_heads[h]["condition"].sum()
            spec_bound = spec_post_heads[h]["condition"].sum()
            exact_bounds.append(exact_bound)
            spec_bounds.append(spec_bound)
            head_error = torch.linalg.vector_norm(post_error)
            ratio = float(head_error / exact_bound.clamp_min(1e-12))
            max_ratio["post_exact_head"] = max(max_ratio["post_exact_head"], ratio)
            if float(head_error) > float(exact_bound) + 1e-9:
                violations["post_exact_head"] += 1
            if float(exact_bound) > float(spec_bound) + 1e-9:
                violations["exact_bound_le_spectral"] += 1

        total_error = torch.linalg.vector_norm(sum(post_errors))
        total_exact_bound = torch.stack(exact_bounds).sum()
        total_spec_bound = torch.stack(spec_bounds).sum()
        for key, bound in (
            ("post_exact_total", total_exact_bound),
            ("post_spectral_total", total_spec_bound),
        ):
            ratio = float(total_error / bound.clamp_min(1e-12))
            max_ratio[key] = max(max_ratio[key], ratio)
            if float(total_error) > float(bound) + 1e-9:
                violations[key] += 1

        hybrid_post_errors = []
        hybrid_certificates = []
        for kv_head in range(n_kv_heads):
            first_h = kv_head * group_size
            n_blocks = len(pre_heads[first_h]["blocks"])
            selected = torch.rand(n_blocks) < 0.35
            for h in range(first_h, first_h + group_size):
                hybrid = _hybrid_output(pre_heads[h], selected)
                hybrid_post_errors.append(
                    w_heads[h] @ (hybrid - pre_heads[h]["exact"])
                )
                unsel = ~selected
                p_hat = exact_post_heads[h]["p_hat"]
                f_value = exact_post_heads[h]["f_value"]
                t_value = (p_hat[unsel] * (f_value[unsel] - 1)).sum()
                value_term = sum(
                    2
                    * p_hat[i]
                    * exact_post_heads[h]["blocks"][i]["b_c"]
                    * torch.tanh(exact_post_heads[h]["blocks"][i]["delta"] / 2)
                    for i in range(n_blocks)
                    if bool(unsel[i])
                )
                certificate = (
                    2
                    * exact_post_heads[h]["b_all"]
                    * t_value
                    / (1 + t_value)
                    + value_term
                )
                hybrid_certificates.append(certificate)

        hybrid_error = torch.linalg.vector_norm(sum(hybrid_post_errors))
        hybrid_bound = torch.stack(hybrid_certificates).sum()
        hybrid_ratio = float(hybrid_error / hybrid_bound.clamp_min(1e-12))
        max_ratio["hybrid_post_exact_total"] = max(
            max_ratio["hybrid_post_exact_total"], hybrid_ratio
        )
        if float(hybrid_error) > float(hybrid_bound) + 1e-9:
            violations["hybrid_post_exact_total"] += 1

    result = {
        "trials": int(args.trials),
        "seed": int(args.seed),
        "violations": violations,
        "max_error_over_bound": max_ratio,
        "passed": all(count == 0 for count in violations.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
