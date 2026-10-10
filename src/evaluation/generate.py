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
    return probs.cpu()


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
