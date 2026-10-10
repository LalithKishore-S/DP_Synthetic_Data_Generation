from __future__ import annotations

import torch

from src.models.vae import VAE


def _batch(n: int = 4) -> torch.Tensor:
    return (torch.rand(n, 1, 28, 28) >= 0.5).float()


def test_encoder_decoder_shapes() -> None:
    model = VAE(latent_dim=16)
    x = _batch()
    mu, logvar = model.encoder(x)
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)
    z = model.reparameterize(mu, logvar)
    logits = model.decoder(z)
    assert logits.shape == (4, 1, 28, 28)


def test_forward_pass() -> None:
    model = VAE(latent_dim=8)
    out = model(_batch(2))
    assert out["logits"].shape == (2, 1, 28, 28)
    assert out["z"].shape == (2, 8)
    assert torch.isfinite(out["logits"]).all()


def test_reparameterization_gradients() -> None:
    model = VAE(latent_dim=8)
    x = _batch(2)
    mu, logvar = model.encoder(x)
    z = model.reparameterize(mu, logvar)
    z.sum().backward()
    assert mu.grad is None  # mu is not a leaf after encoder; check encoder params
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters() if p.requires_grad]
    assert any(grads)
