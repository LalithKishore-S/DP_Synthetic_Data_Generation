"""MNIST protocol from Section V-A of papers/DP_VAE.pdf.

VERIFIED: 9999 uniform train samples + one all-white outlier; binarise to {0,1}.
ASSUMED: hard threshold at `binarize_threshold` after scaling to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Binarize:
    """Map pixels in [0, 1] to {0, 1}. ASSUMED threshold rule."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return binarize(tensor, self.threshold)


def binarize(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (tensor >= threshold).to(dtype=torch.float32)


def white_outlier(size: int = 28) -> torch.Tensor:
    """All-white image with pixel value 1 (Section V, p. 6)."""
    return torch.ones(1, size, size, dtype=torch.float32)


@dataclass
class MNISTPaperSplit:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    train_indices: np.ndarray
    test_images: torch.Tensor
    test_labels: torch.Tensor
    outlier: torch.Tensor


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def _load_mnist_tensors(data_dir: str | Path, download: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    train = datasets.MNIST(root=str(data_dir), train=True, download=download, transform=transforms.ToTensor())
    test = datasets.MNIST(root=str(data_dir), train=False, download=download, transform=transforms.ToTensor())
    train_x = train.data.unsqueeze(1).float() / 255.0
    test_x = test.data.unsqueeze(1).float() / 255.0
    return train_x, train.targets.clone(), test_x, test.targets.clone()


def build_mnist_paper_data(
    data_dir: str | Path,
    n_train: int = 9999,
    binarize_threshold: float = 0.5,
    add_white_outlier: bool = True,
    seed: int = 42,
    download: bool = True,
) -> MNISTPaperSplit:
    """Build the Section V-A training set and keep full test tensors for MIA."""
    train_x, train_y, test_x, test_y = _load_mnist_tensors(data_dir, download=download)
    train_x = binarize(train_x, binarize_threshold)
    test_x = binarize(test_x, binarize_threshold)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_x.shape[0], size=n_train, replace=False)
    chosen.sort()
    subset_x = train_x[chosen]
    subset_y = train_y[chosen]

    outlier = white_outlier(size=int(train_x.shape[-1]))
    if add_white_outlier:
        subset_x = torch.cat([subset_x, outlier.unsqueeze(0)], dim=0)
        subset_y = torch.cat([subset_y, torch.tensor([-1], dtype=torch.long)], dim=0)

    return MNISTPaperSplit(
        train_images=subset_x,
        train_labels=subset_y,
        train_indices=chosen,
        test_images=test_x,
        test_labels=test_y,
        outlier=outlier,
    )


class MNISTPaperDataset(TensorImageDataset):
    """Convenience dataset over the paper training images."""


def make_loader(images: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    dataset = TensorImageDataset(images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def sample_mia_sets(
    split: MNISTPaperSplit,
    n_members: int,
    n_nonmembers: int,
    seed: int,
    include_outlier_as_member: bool = True,
) -> dict[str, Any]:
    """1k members from training, 1k non-members from MNIST test (Section V-A)."""
    rng = np.random.default_rng(seed)
    n_train = int(split.train_images.shape[0])
    member_pool = np.arange(n_train)
    if n_members > n_train:
        raise ValueError("n_members exceeds training set size")
    members = rng.choice(member_pool, size=n_members, replace=False)
    if include_outlier_as_member and n_train > 0:
        outlier_index = n_train - 1
        if outlier_index not in members:
            members[0] = outlier_index

    n_test = int(split.test_images.shape[0])
    nonmembers = rng.choice(n_test, size=n_nonmembers, replace=False)
    return {
        "member_indices": members,
        "nonmember_indices": nonmembers,
        "member_images": split.train_images[members],
        "nonmember_images": split.test_images[nonmembers],
    }


# Keep torchvision Subset import used for type compatibility in tests.
__all__ = [
    "Binarize",
    "MNISTPaperDataset",
    "MNISTPaperSplit",
    "Subset",
    "binarize",
    "build_mnist_paper_data",
    "make_loader",
    "sample_mia_sets",
    "white_outlier",
]
