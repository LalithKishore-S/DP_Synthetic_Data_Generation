from __future__ import annotations

import torch

from src.data.mnist import binarize, sample_mia_sets, white_outlier
from src.data.mnist import MNISTPaperSplit


def test_binarize_threshold() -> None:
    x = torch.tensor([0.0, 0.49, 0.5, 1.0])
    y = binarize(x, threshold=0.5)
    assert torch.equal(y, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_white_outlier_shape_and_value() -> None:
    img = white_outlier(28)
    assert img.shape == (1, 28, 28)
    assert torch.all(img == 1.0)


def test_mia_split_sizes() -> None:
    train = torch.zeros(20, 1, 28, 28)
    train[-1] = 1.0
    split = MNISTPaperSplit(
        train_images=train,
        train_labels=torch.zeros(20, dtype=torch.long),
        train_indices=torch.arange(19).numpy(),
        test_images=torch.zeros(30, 1, 28, 28),
        test_labels=torch.zeros(30, dtype=torch.long),
        outlier=white_outlier(),
    )
    sets = sample_mia_sets(split, n_members=5, n_nonmembers=7, seed=0)
    assert sets["member_images"].shape[0] == 5
    assert sets["nonmember_images"].shape[0] == 7
    assert 19 in set(sets["member_indices"].tolist())

def test_build_paper_data_has_exactly_one_white_outlier(monkeypatch) -> None:
    import src.data.mnist as m
    monkeypatch.setattr(
        m, "_load_mnist_tensors",
        lambda d, download: (torch.rand(200, 1, 28, 28), torch.zeros(200, dtype=torch.long),
                             torch.rand(50, 1, 28, 28), torch.zeros(50, dtype=torch.long)),
    )
    split = m.build_mnist_paper_data("unused", n_train=99, seed=0)
    assert split.train_images.shape[0] == 100
    white = (split.train_images.flatten(1) == 1).all(dim=1)
    assert white.sum() == 1 and bool(white[-1])