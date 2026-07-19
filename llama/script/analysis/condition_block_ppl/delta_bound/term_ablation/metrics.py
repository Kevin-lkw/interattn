"""Ranking and hybrid-selection metrics used by the term ablation."""

from dataclasses import dataclass
import math

import torch


def rankdata(values):
    """Average ranks for a 1-D tensor, including ties."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    i = 0
    while i < values.numel():
        j = i + 1
        while j < values.numel() and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson(x, y):
    if x.numel() < 2:
        return float("nan")
    x = x.double()
    y = y.double()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)
    if float(denom) == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman(score, target):
    valid = torch.isfinite(score) & torch.isfinite(target)
    score = score[valid]
    target = target[valid]
    if score.numel() < 2:
        return float("nan")
    return pearson(rankdata(score), rankdata(target))


def topk_mask(score, k):
    k = max(1, min(int(k), int(score.numel())))
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask[torch.argsort(score, descending=True)[:k]] = True
    return mask


def topk_metrics(score, target, k):
    """Overlap, positive-target capture, and NDCG for a same-size top-k set."""
    k = max(1, min(int(k), int(score.numel())))
    predicted = torch.argsort(score, descending=True)[:k]
    ideal = torch.argsort(target, descending=True)[:k]
    overlap = len(set(predicted.tolist()).intersection(ideal.tolist())) / k

    relevance = target.double().clamp_min(0.0)
    ideal_sum = relevance[ideal].sum()
    capture = (
        float(relevance[predicted].sum() / ideal_sum)
        if float(ideal_sum) > 0.0
        else float("nan")
    )
    discount = torch.log2(torch.arange(k, dtype=torch.float64) + 2.0).to(target.device)
    dcg = (relevance[predicted] / discount).sum()
    idcg = (relevance[ideal] / discount).sum()
    ndcg = float(dcg / idcg) if float(idcg) > 0.0 else float("nan")
    return {"overlap": overlap, "capture": capture, "ndcg": ndcg}


@dataclass
class HybridState:
    """Stable exact/approximate block numerators for one head and query."""

    approx_num: torch.Tensor
    approx_den: torch.Tensor
    exact_num: torch.Tensor
    exact_den: torch.Tensor
    full_output: torch.Tensor
    post_gram: torch.Tensor | None = None

    @property
    def num_blocks(self):
        return int(self.approx_den.numel())

    def _norms(self, output_delta):
        pre = torch.linalg.vector_norm(output_delta, dim=-1)
        if self.post_gram is None:
            post = torch.full_like(pre, float("nan"))
        else:
            post_sq = torch.einsum(
                "...d,de,...e->...", output_delta, self.post_gram, output_delta
            )
            post = post_sq.clamp_min(0.0).sqrt()
        return pre, post

    def error(self, selected):
        selected = selected.to(self.approx_den.device, dtype=torch.bool)
        num = torch.where(
            selected[:, None], self.exact_num, self.approx_num
        ).sum(dim=0)
        den = torch.where(selected, self.exact_den, self.approx_den).sum()
        pre, post = self._norms(num / den - self.full_output)
        return float(pre), float(post)

    def candidate_errors(self, selected=None):
        """Error after adding each not-yet-selected block to the exact set."""
        if selected is None:
            selected = torch.zeros(
                self.num_blocks, device=self.approx_den.device, dtype=torch.bool
            )
        selected = selected.to(self.approx_den.device, dtype=torch.bool)
        num = torch.where(
            selected[:, None], self.exact_num, self.approx_num
        ).sum(dim=0)
        den = torch.where(selected, self.exact_den, self.approx_den).sum()
        candidate_num = num[None, :] + self.exact_num - self.approx_num
        candidate_den = den + self.exact_den - self.approx_den
        delta = candidate_num / candidate_den[:, None] - self.full_output[None, :]
        pre, post = self._norms(delta)
        pre = pre.masked_fill(selected, float("inf"))
        post = post.masked_fill(selected, float("inf"))
        return pre, post

    def single_block_gains(self):
        empty = torch.zeros(
            self.num_blocks, device=self.approx_den.device, dtype=torch.bool
        )
        base_pre, base_post = self.error(empty)
        candidate_pre, candidate_post = self.candidate_errors(empty)
        gain_pre = base_pre - candidate_pre
        gain_post = base_post - candidate_post
        return gain_pre, gain_post

    def greedy_curve(self, ks, objective="pre"):
        """Greedy marginal oracle; useful as a selection ceiling, not a bound."""
        if objective not in {"pre", "post"}:
            raise ValueError(f"Unknown greedy objective: {objective}")
        if objective == "post" and self.post_gram is None:
            raise ValueError("post objective requires post_gram")
        wanted = set(int(k) for k in ks)
        selected = torch.zeros(
            self.num_blocks, device=self.approx_den.device, dtype=torch.bool
        )
        curve = {}
        for step in range(1, max(wanted) + 1):
            pre, post = self.candidate_errors(selected)
            objective_error = pre if objective == "pre" else post
            selected[int(torch.argmin(objective_error))] = True
            if step in wanted:
                err_pre, err_post = self.error(selected)
                curve[step] = {
                    "pre_error": err_pre,
                    "post_error": err_post,
                }
        return curve


def finite_mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def finite_median(values):
    values = torch.tensor(
        [float(v) for v in values if math.isfinite(float(v))], dtype=torch.float64
    )
    return float(values.median()) if values.numel() else float("nan")
