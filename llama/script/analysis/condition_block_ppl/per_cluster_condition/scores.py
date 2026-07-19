"""Score and certificate formulas for the additive per-cluster condition."""

import torch


def condition_scores(p_hat, delta, b_c, b_all):
    """Return comparable block scores in float64 for numerical headroom."""
    p_hat = p_hat.double()
    delta = delta.double().clamp_min(0.0)
    b_c = b_c.double()
    cosh_delta = torch.cosh(delta)
    tanh_half = torch.tanh(delta / 2.0)
    s_delta = (p_hat * cosh_delta).sum().clamp_min(1e-300)

    original_mass = p_hat * 2.0 * b_all * (cosh_delta - 1.0) / s_delta
    per_cluster_mass = p_hat * 2.0 * b_all * (cosh_delta - 1.0)
    value = p_hat * 2.0 * b_c * tanh_half
    mass_exp = 2.0 * b_all * torch.softmax(
        torch.log(p_hat.clamp_min(1e-300)) + delta,
        dim=0,
    )
    return {
        "original": original_mass + value,
        "per_cluster": per_cluster_mass + value,
        "term1": original_mass,
        "mass_exp": mass_exp + value,
    }, {
        "s_delta": s_delta,
        "original_mass": original_mass,
        "per_cluster_mass": per_cluster_mass,
        "value": value,
    }


def per_cluster_condition_score(
    *,
    p_tensor,
    z_logits,
    delta,
    b_c,
    b_all,
    cluster_exists,
):
    """Runner hook for p_hat[2B(cosh(delta)-1) + value term]."""
    del z_logits
    p_tensor = p_tensor.double()
    delta = delta.double().clamp_min(0.0)
    b_c = b_c.double()
    score = p_tensor * (
        2.0 * b_all.double().unsqueeze(-1) * (torch.cosh(delta) - 1.0)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    return score.masked_fill(~cluster_exists.unsqueeze(0), 0.0)


def hybrid_certificates(p_hat, delta, b_c, b_all, unselected):
    """Return tight set-wise and additive per-cluster hybrid certificates."""
    p_hat = p_hat.double()
    delta = delta.double().clamp_min(0.0)
    b_c = b_c.double()
    unselected = unselected.bool()
    mass_units = p_hat * (torch.cosh(delta) - 1.0)
    value = p_hat * 2.0 * b_c * torch.tanh(delta / 2.0)
    t_value = mass_units[unselected].sum()
    tight = 2.0 * b_all * t_value / (1.0 + t_value) + value[unselected].sum()
    additive = 2.0 * b_all * mass_units[unselected].sum() + value[unselected].sum()
    return float(tight), float(additive)
