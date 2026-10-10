"""ELBO terms (Section III-A) and decoder gradient penalty (Section V-A / [24])."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def reconstruction_bce(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits. Sum over pixels, mean over batch.

    VERIFIED: BCE reconstruction on binarised MNIST (Section V-A).
    INFERRED: logits form because the decoder has no output activation.
    """
    per_example = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return per_example.flatten(1).sum(dim=1).mean()


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(q(z|x) || N(0,I)). INFERRED from [9]."""
    per_example = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_example.mean()


def elbo_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1.0,
) -> dict[str, torch.Tensor]:
    recon = reconstruction_bce(logits, x)
    kl = kl_standard_normal(mu, logvar)
    total = recon + beta_kl * kl
    return {"loss": total, "recon": recon, "kl": kl}


def decoder_gradient_penalty(
    decoder: nn.Module,
    z: torch.Tensor,
    lipschitz_L: float,
) -> torch.Tensor:
    """WGAN-GP-style penalty on ||∂μ_θ/∂z|| (ASSUMED adaptation of [24]).

    Uses a random output projection so the penalty is a scalar function of z.
    """
    z_in = z.detach().requires_grad_(True)
    output = decoder(z_in)
    noise = torch.randn_like(output)
    flat = noise.flatten(1)
    noise = noise / (flat.norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=z_in,
        grad_outputs=noise,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grads.flatten(1).norm(dim=1)
    return ((grad_norm - lipschitz_L) ** 2).mean()
