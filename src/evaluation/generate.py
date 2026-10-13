"""Ancestral sampling: z ~ N(0, I), decode (Section III-A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.vae import VAE


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


@torch.no_grad()
def generate_samples(
    model: VAE,
    n: int,
    device: torch.device | None = None,
    postprocess: str = "sigmoid",
) -> torch.Tensor:
    model.eval()
    logits = model.sample(n, device=device)
    if postprocess == "logits":
        return logits.cpu()
    probs = logits_to_probs(logits)
    if postprocess == "threshold":
        return (probs >= 0.5).float().cpu()
    if postprocess == "bernoulli":
        return torch.bernoulli(probs).cpu()
    return probs.cpu()


@torch.no_grad()
def generate_samples_batched(
    model: VAE,
    n: int,
    device: torch.device | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Sigmoid samples like generate_samples(..., postprocess="sigmoid"), decoded in chunks.

    The latent draw is the SAME single `torch.randn(n, latent_dim)` call that
    VAE.sample makes (only n x latent_dim floats), so the global RNG stream and
    the samples match generate_samples; only the decoder is applied in batches,
    which avoids materialising decoder activations for all n samples at once.
    """
    model.eval()
    device = device or next(model.parameters()).device
    z = torch.randn(n, model.latent_dim, device=device)
    out: torch.Tensor | None = None
    for i in range(0, n, batch_size):
        probs = logits_to_probs(model.decoder(z[i : i + batch_size])).cpu()
        if out is None:
            out = torch.empty((n, *probs.shape[1:]), dtype=probs.dtype)
        out[i : i + probs.shape[0]] = probs
    assert out is not None
    return out


@torch.no_grad()
def reconstruct(model: VAE, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    out = model(images.to(device))
    return logits_to_probs(out["logits"]).cpu()


def reconstruction_grid(model: VAE, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    recon = reconstruct(model, images, device)
    return torch.cat([images.cpu(), recon], dim=0)


def save_samples_npy(samples: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, samples.numpy())
