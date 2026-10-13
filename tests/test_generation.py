from __future__ import annotations

import torch

from src.evaluation.generate import generate_samples
from src.evaluation.mia import monte_carlo_nearest_attack
from src.models.vae import VAE


def test_synthetic_generation_shape() -> None:
    model = VAE(latent_dim=8)
    samples = generate_samples(model, n=6, device=torch.device("cpu"))
    assert samples.shape == (6, 1, 28, 28)
    assert torch.isfinite(samples).all()
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_mia_smoke_curve() -> None:
    members = torch.zeros(16, 1, 28, 28).numpy()
    nonmembers = torch.ones(16, 1, 28, 28).numpy()
    synth = torch.zeros(32, 1, 28, 28).numpy()
    result = monte_carlo_nearest_attack(members, nonmembers, synth, n_thresholds=20)
    assert result["type_i"].shape == (20,)
    assert result["type_ii"].shape == (20,)
    assert result["member_scores"].mean() < result["nonmember_scores"].mean()
