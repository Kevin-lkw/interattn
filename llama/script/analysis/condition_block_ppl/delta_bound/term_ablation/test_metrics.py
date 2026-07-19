import torch

from .metrics import HybridState, topk_metrics
from .run import compute_scores


def test_topk_metrics_perfect_ranking():
    score = torch.tensor([3.0, 1.0, 2.0])
    metrics = topk_metrics(score, score, 2)
    assert metrics == {"overlap": 1.0, "capture": 1.0, "ndcg": 1.0}


def test_hybrid_state_is_exact_when_all_blocks_selected():
    exact_num = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    exact_den = torch.tensor([2.0, 2.0], dtype=torch.float64)
    state = HybridState(
        approx_num=torch.tensor([[1.0, 0.0], [0.0, 1.5]], dtype=torch.float64),
        approx_den=torch.tensor([1.0, 1.0], dtype=torch.float64),
        exact_num=exact_num,
        exact_den=exact_den,
        full_output=exact_num.sum(dim=0) / exact_den.sum(),
        post_gram=torch.eye(2, dtype=torch.float64),
    )
    pre, post = state.error(torch.ones(2, dtype=torch.bool))
    assert pre == 0.0
    assert post == 0.0


def test_term1_log_exact_has_identical_order():
    p_hat = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    delta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b_c = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
    scores, _ = compute_scores(p_hat, delta, b_c, b_all=4.0)
    assert torch.equal(
        torch.argsort(scores["term1"]),
        torch.argsort(scores["term1_log_exact"]),
    )


def test_mass_exp_matches_large_delta_full_mass():
    p_hat = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    delta = torch.tensor([20.0, 22.0, 25.0], dtype=torch.float64)
    b_c = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
    scores, _ = compute_scores(p_hat, delta, b_c, b_all=4.0)
    assert torch.allclose(
        scores["full_no_minus1"], scores["full_mass_exp"], rtol=1e-8, atol=1e-10
    )
