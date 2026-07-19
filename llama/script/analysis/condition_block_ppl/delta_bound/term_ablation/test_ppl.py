import torch

from ... import runner_cond_block as rcb
from .ppl import condition_score
from .summarize_ppl import interpolate_curve


def _inputs():
    z_logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.5, -0.5, float("-inf")]]],
        dtype=torch.float64,
    )
    cluster_exists = torch.isfinite(z_logits[0])
    p_tensor = torch.softmax(z_logits, dim=-1)
    delta = torch.tensor(
        [[[0.2, 1.0, 2.0], [0.4, 1.5, 0.0]]], dtype=torch.float64
    )
    b_c = torch.tensor(
        [[[1.0, 2.0, 1.5], [0.8, 1.2, 0.0]]], dtype=torch.float64
    )
    b_all = b_c.amax(dim=-1)
    return p_tensor, z_logits, delta, b_c, b_all, cluster_exists


def test_original_score_hook_is_identical_to_runner_default():
    p_tensor, z_logits, delta, b_c, b_all, cluster_exists = _inputs()
    kwargs = {
        "p_tensor": p_tensor,
        "z_logits": z_logits,
        "delta": delta,
        "b_c": b_c,
        "b_all": b_all,
        "cluster_exists": cluster_exists,
    }
    expected = rcb._condition_score_for_blocks(**kwargs)
    actual = condition_score(
        **kwargs, score_kind="original", term2_weight=1.0
    )
    assert torch.equal(actual, expected)


def test_mass_exp_upper_bounds_original_term1():
    p_tensor, z_logits, delta, b_c, b_all, cluster_exists = _inputs()
    zeros = torch.zeros_like(b_c)
    original = condition_score(
        p_tensor=p_tensor,
        z_logits=z_logits,
        delta=delta,
        b_c=zeros,
        b_all=b_all,
        cluster_exists=cluster_exists,
        score_kind="original",
        term2_weight=0.0,
    )
    mass_exp = condition_score(
        p_tensor=p_tensor,
        z_logits=z_logits,
        delta=delta,
        b_c=zeros,
        b_all=b_all,
        cluster_exists=cluster_exists,
        score_kind="mass_exp",
        term2_weight=0.0,
    )
    assert torch.all(mass_exp >= original)


def test_interpolate_curve():
    points = [(0.1, 3.0), (0.3, 2.0), (0.5, 1.5)]
    assert interpolate_curve(points, 0.1) == 3.0
    assert interpolate_curve(points, 0.2) == 2.5
    assert interpolate_curve(points, 0.5) == 1.5
