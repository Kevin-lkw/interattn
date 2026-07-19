import torch

from .scores import condition_scores, hybrid_certificates


def test_per_cluster_mass_is_s_delta_times_original_mass():
    p_hat = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    delta = torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64)
    b_c = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)
    _scores, parts = condition_scores(p_hat, delta, b_c, b_all=3.0)
    assert torch.allclose(
        parts["per_cluster_mass"],
        parts["s_delta"] * parts["original_mass"],
    )


def test_additive_certificate_is_no_smaller_than_tight_certificate():
    p_hat = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    delta = torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64)
    b_c = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)
    tight, additive = hybrid_certificates(
        p_hat,
        delta,
        b_c,
        b_all=3.0,
        unselected=torch.tensor([True, False, True]),
    )
    assert additive >= tight
