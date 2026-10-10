"""Convolutional encoder q_phi(z|x). Section V-A of papers/DP_VAE.pdf."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.padding import pad_same


class Encoder(nn.Module):
    """4 conv layers (32, 64, 128, 32), 3x3, stride 2, then dense 100 and 2*n_latent.

    Hidden activations: ReLU. Output: no activation (VERIFIED).
    The 2*n_latent vector is split into mu and log variance (ASSUMED log σ²).
    """

    def __init__(
        self,
        latent_dim: int,
        conv_filters: Sequence[int] = (32, 64, 128, 32),
        kernel_size: int = 3,
        stride: int = 2,
        hidden: int = 100,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        layers: list[nn.Module] = []
        prev = in_channels
        for filters in conv_filters:
            layers.append(nn.Conv2d(prev, filters, kernel_size=kernel_size, stride=stride, padding=0))
            layers.append(nn.ReLU())
            prev = filters
        self.conv_layers = nn.ModuleList(layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 28, 28)
            flat = self._convolve(dummy).flatten(1).shape[1]
        self.fc_hidden = nn.Linear(flat, hidden)
        self.fc_out = nn.Linear(hidden, 2 * latent_dim)

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.conv_layers:
            if isinstance(module, nn.Conv2d):
                x = pad_same(x, self.kernel_size, self.stride)
                x = module(x)
            else:
                x = module(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._convolve(x)
        h = torch.relu(self.fc_hidden(h.flatten(1)))
        stats = self.fc_out(h)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, logvar
