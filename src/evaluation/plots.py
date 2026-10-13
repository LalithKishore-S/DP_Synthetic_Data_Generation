"""Plot helpers. All files go under results/<experiment_id>/plots/."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pics = images.detach().cpu().clamp(0, 1)
    n = pics.shape[0]
    nrow = max(1, min(nrow, n))
    ncol = int(np.ceil(n / nrow))
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    axes_list = np.atleast_1d(axes).ravel()
    for ax in axes_list:
        ax.axis("off")
    for i in range(n):
        axes_list[i].imshow(pics[i].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_training_plot(history: Sequence[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("loss", "recon", "kl"):
        ax.plot(epochs, [row[key] for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.legend()
    ax.set_title("Training (OUR RESULT)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_mia_tradeoff_plot(type_i: np.ndarray, type_ii: np.ndarray, path: Path, title: str = "MIA trade-off (OUR RESULT)") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(type_i, type_ii, label="attack")
    ax.plot([0, 1], [1, 0], linestyle="--", color="gray", label="random")
    ax.set_xlabel("Type I (false alarm)")
    ax.set_ylabel("Type II (missed detection)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
