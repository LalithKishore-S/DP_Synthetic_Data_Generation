from __future__ import annotations

import torch

from src.models.losses import elbo_loss, kl_standard_normal, reconstruction_bce


def test_kl_zero_for_standard_normal() -> None:
    mu = torch.zeros(3, 5)
    logvar = torch.zeros(3, 5)
    kl = kl_standard_normal(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_known_mean_shift() -> None:
    d = 4
    mu = torch.ones(2, d)
    logvar = torch.zeros(2, d)
    kl = kl_standard_normal(mu, logvar)
    expected = 0.5 * d
    assert torch.isclose(kl, torch.tensor(expected), atol=1e-5)


def test_reconstruction_nonnegative() -> None:
    x = torch.zeros(2, 1, 28, 28)
    logits = torch.zeros_like(x)
    recon = reconstruction_bce(logits, x)
    assert recon.ndim == 0
    assert recon.item() >= 0


def test_elbo_finite() -> None:
    x = (torch.rand(2, 1, 28, 28) >= 0.5).float()
    logits = torch.zeros_like(x)
    mu = torch.zeros(2, 8)
    logvar = torch.zeros(2, 8)
    terms = elbo_loss(logits, x, mu, logvar)
    assert torch.isfinite(terms["loss"])
    assert terms["recon"].item() >= 0
    assert terms["kl"].item() >= -1e-6
