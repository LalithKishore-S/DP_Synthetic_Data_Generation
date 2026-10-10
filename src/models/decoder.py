"""Decoder μ_θ(z). Section V-A; last-layer stride is ASSUMED (see methodology)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    return nn.utils.parametrizations.spectral_norm(module)


class Decoder(nn.Module):
    """Dense 100 → 7*7*32, then TConv 64 and 128 (stride 2), then 1-channel conv.

    ASSUMED: the paper lists three stride-2 transposed convs, which would map
    7x7 → 56x56. We use two stride-2 TConvs (7→14→28) and a stride-1 conv to
    1 channel so reconstructions match MNIST spatial size.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 100,
        reshape: Sequence[int] = (32, 7, 7),
        tconv_filters: Sequence[int] = (64, 128),
        kernel_size: int = 3,
        spectral_norm: bool = False,
        lipschitz_L: float = 1.0,
    ) -> None:
        super().__init__()
        c0, h0, w0 = reshape
        self.reshape = (c0, h0, w0)
        self.spectral_norm = spectral_norm
        n_weight_layers = 2 + len(tconv_filters) + 1
        self.layer_scale = float(lipschitz_L) ** (1.0 / n_weight_layers) if spectral_norm else 1.0

        self.fc1 = _maybe_spectral_norm(nn.Linear(latent_dim, hidden), spectral_norm)
        self.fc2 = _maybe_spectral_norm(nn.Linear(hidden, c0 * h0 * w0), spectral_norm)

        tconvs: list[nn.Module] = []
        prev = c0
        for filters in tconv_filters:
            tconvs.append(
                _maybe_spectral_norm(
                    nn.ConvTranspose2d(
                        prev,
                        filters,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    spectral_norm,
                )
            )
            prev = filters
        self.tconvs = nn.ModuleList(tconvs)
        self.out_conv = _maybe_spectral_norm(
            nn.Conv2d(prev, 1, kernel_size=kernel_size, stride=1, padding=1),
            spectral_norm,
        )

    def _scale(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.layer_scale == 1.0:
            return tensor
        return tensor * self.layer_scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self._scale(self.fc1(z)))
        h = torch.relu(self._scale(self.fc2(h)))
        c0, h0, w0 = self.reshape
        h = h.view(z.shape[0], c0, h0, w0)
        for tconv in self.tconvs:
            h = torch.relu(self._scale(tconv(h)))
        return self._scale(self.out_conv(h))
