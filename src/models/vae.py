"""VAE with reparameterization (Kingma & Welling [9]; Section III-A)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        conv_filters: Sequence[int] = (32, 64, 128, 32),
        encoder_hidden: int = 100,
        decoder_hidden: int = 100,
        decoder_reshape: Sequence[int] = (32, 7, 7),
        tconv_filters: Sequence[int] = (64, 128),
        spectral_norm: bool = False,
        lipschitz_L: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            latent_dim=latent_dim,
            conv_filters=conv_filters,
            hidden=encoder_hidden,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden=decoder_hidden,
            reshape=decoder_reshape,
            tconv_filters=tconv_filters,
            spectral_norm=spectral_norm,
            lipschitz_L=lipschitz_L,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)
