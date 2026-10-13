from __future__ import annotations

import torch

from src.models.losses import decoder_gradient_penalty, elbo_loss
from src.models.vae import VAE


def test_elbo_backward_all_params() -> None:
    model = VAE(latent_dim=8)
    x = (torch.rand(3, 1, 28, 28) >= 0.5).float()
    out = model(x)
    terms = elbo_loss(out["logits"], x, out["mu"], out["logvar"])
    terms["loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


# def test_gradient_penalty_backward() -> None:
#     model = VAE(latent_dim=8)
#     z = torch.randn(2, 8)
#     gp = decoder_gradient_penalty(model.decoder, z, lipschitz_L=1.0)
#     gp.backward()
#     decoder_grads = [p.grad is not None for n, p in model.named_parameters() if n.startswith("decoder.") and p.requires_grad]
#     assert any(decoder_grads)
#     assert torch.isfinite(gp)

def test_gradient_penalty_backward() -> None:
    model = VAE(latent_dim=8)
    z = torch.randn(4, 8)
    gp = decoder_gradient_penalty(model.decoder, z, lipschitz_L=1e-3)
    assert torch.isfinite(gp) and gp.item() > 0
    gp.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if n.startswith("decoder.")
    )
