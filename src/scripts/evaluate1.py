#!/usr/bin/env python3
"""
Paper-aligned evaluation for:
Groß & Wunder (2023), "Differentially Private Synthetic Data Generation
via Lipschitz-Regularised Variational Autoencoders", Section V.

Paper protocol:
- MNIST pixels binarised to {0, 1}.
- Training data: 9,999 randomly selected MNIST training examples + one all-white image.
- MIA: 1,000 training members and 1,000 MNIST test-set non-members.
- Monte Carlo MIA: nearest-synthetic-record distance, threshold swept to obtain
  Type-I false-alarm vs Type-II missed-detection trade-off.
- Save generated samples and the reconstruction of the all-white outlier.

IMPORTANT:
The training dataset used here must match the dataset used to train the checkpoint
(including the random seed and selected MNIST training records). For strict paper
reproduction, set dataset.n_train=9999 and dataset.add_white_outlier=true in the
checkpoint config, and train with that same configuration.

The paper does not clearly specify the number of synthetic records used for the
Monte Carlo attack. This script defaults to eval.n_synthetic from the config, or
10,000 if that setting is absent. Set --n-synthetic to control it explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import pairwise_distances, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.mnist import build_mnist_paper_data, sample_mia_sets
from src.evaluation.generate import (
    logits_to_probs,
    reconstruct,
    save_samples_npy,
)
from src.evaluation.plots import save_image_grid, save_mia_tradeoff_plot
from src.training.trainer import load_checkpoint
from src.utils.device import get_device
from src.utils.experiment import create_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-aligned VAE/LVAE evaluation (MNIST, Section V)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, default="paper_eval")
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=None,
        help="Number of synthetic records used by MIA (default: config value or 10000).",
    )
    parser.add_argument("--n-members", type=int, default=1000)
    parser.add_argument("--n-nonmembers", type=int, default=1000)
    parser.add_argument("--n-sample-grid", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mia-input",
        choices=("bernoulli", "probabilities"),
        default="bernoulli",
        help=(
            "Use sampled binary pixels from the decoder's Bernoulli likelihood "
            "for MIA (paper's probabilistic interpretation), or use decoder "
            "probabilities as continuous synthetic records."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small quick run for checking the evaluation pipeline only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_flat_matrix(images: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    array = np.asarray(images, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"Expected a batch of images, got shape {array.shape}")
    return array.reshape(array.shape[0], -1)


def nearest_synthetic_distances(
    targets: np.ndarray | torch.Tensor,
    synthetics: np.ndarray | torch.Tensor,
    chunk_size: int = 256,
) -> np.ndarray:
    """Euclidean distance from each target to its nearest synthetic record."""
    target_matrix = as_flat_matrix(targets)
    synthetic_matrix = as_flat_matrix(synthetics)

    if len(synthetic_matrix) == 0:
        raise ValueError("The synthetic dataset is empty.")

    # Chunk target records to avoid allocating one enormous distance matrix.
    nearest = []
    for start in range(0, len(target_matrix), chunk_size):
        block = target_matrix[start : start + chunk_size]
        distances = pairwise_distances(
            block, synthetic_matrix, metric="euclidean"
        )
        nearest.append(distances.min(axis=1))
    return np.concatenate(nearest)


def paper_tradeoff_curve(
    member_distances: np.ndarray,
    nonmember_distances: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Threshold attack: predict member iff nearest-synthetic distance < tau.
    Type I = non-member falsely called a member.
    Type II = member missed by the attack.

    Include +/- infinity so the curve contains both exact endpoints.
    """
    all_scores = np.concatenate([member_distances, nonmember_distances])
    thresholds = np.concatenate(
        (
            np.array([-np.inf]),
            np.unique(all_scores),
            np.array([np.inf]),
        )
    )

    type_i = np.empty(len(thresholds), dtype=np.float64)
    type_ii = np.empty(len(thresholds), dtype=np.float64)

    for i, tau in enumerate(thresholds):
        type_i[i] = np.mean(nonmember_distances < tau)
        type_ii[i] = np.mean(member_distances >= tau)

    return {
        "threshold": thresholds,
        "type_i": type_i,
        "type_ii": type_ii,
    }


@torch.no_grad()
def draw_decoder_probabilities(
    model: Any,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Draw z from the model's prior through model.sample(), then convert decoder
    logits to Bernoulli pixel probabilities. Returns probabilities in [0, 1].
    """
    model.eval()
    logits = model.sample(n, device=device)
    probabilities = logits_to_probs(logits)
    return probabilities.detach().cpu()


def main() -> None:
    args = parse_args()
    device = get_device()

    model, config = load_checkpoint(args.checkpoint, device=device)

    experiment = config.get("experiment", {})
    results_root = experiment.get("results_root", "results")
    experiment_dir = create_experiment_dir(results_root, args.experiment_id)

    dataset_config = config["dataset"]
    train_config = config.get("train", {})
    eval_config = config.get("eval", {})

    seed = (
        args.seed
        if args.seed is not None
        else int(train_config.get("seed", 42))
    )
    set_seed(seed)

    n_synthetic = args.n_synthetic
    if n_synthetic is None:
        n_synthetic = int(eval_config.get("n_synthetic", 10_000))

    n_members = args.n_members
    n_nonmembers = args.n_nonmembers
    n_sample_grid = args.n_sample_grid

    if args.smoke:
        n_synthetic = min(n_synthetic, 128)
        n_members = min(n_members, 32)
        n_nonmembers = min(n_nonmembers, 32)
        n_sample_grid = min(n_sample_grid, 32)

    configured_n_train = int(dataset_config.get("n_train", 9999))
    configured_outlier = bool(dataset_config.get("add_white_outlier", True))

    if configured_n_train != 9999:
        print(
            "[WARNING] Paper uses 9,999 MNIST training images plus the white "
            f"outlier; config says n_train={configured_n_train}. "
            "For a faithful reproduction, retrain using n_train=9999."
        )
    if not configured_outlier:
        print(
            "[WARNING] config.dataset.add_white_outlier is false. The paper "
            "includes the all-white outlier in training."
        )

    # Recreate the exact training/test data split from the saved config and seed.
    # This is important: member examples must come from the actual training subset.
    split = build_mnist_paper_data(
        data_dir=dataset_config["data_dir"],
        n_train=configured_n_train,
        binarize_threshold=float(dataset_config.get("binarize_threshold", 0.5)),
        add_white_outlier=configured_outlier,
        seed=seed,
    )

    # Generate decoder means/probabilities for visual inspection. These are
    # easier to compare with the grayscale sample grids shown in the paper.
    probabilities = draw_decoder_probabilities(model, n_synthetic, device)

    # For the Monte Carlo attack, use samples from the decoder likelihood by
    # default: x_j ~ Bernoulli(sigmoid(logits_j)). Use --mia-input probabilities
    # to test the alternative continuous-distance convention.
    if args.mia_input == "bernoulli":
        mia_synthetics = torch.bernoulli(probabilities).float()
    else:
        mia_synthetics = probabilities

    save_samples_npy(
        probabilities,
        experiment_dir / "samples" / "generated_probabilities.npy",
    )
    save_samples_npy(
        mia_synthetics,
        experiment_dir / "samples" / "synthetic_records_used_for_mia.npy",
    )
    save_image_grid(
        probabilities[:n_sample_grid],
        experiment_dir / "plots" / "generated_samples.png",
        nrow=8,
        title="Generated samples (decoder probabilities)",
    )

    # Paper's outlier figure is an encoder -> decoder reconstruction, not a
    # random ancestral sample. The input is the all-white training outlier.
    outlier = split.outlier.unsqueeze(0).float()
    outlier_reconstruction = reconstruct(model, outlier, device)

    save_image_grid(
        torch.cat([outlier.cpu(), outlier_reconstruction], dim=0),
        experiment_dir / "plots" / "outlier_reconstruction.png",
        nrow=2,
        title="White outlier: input | reconstruction",
    )
    save_samples_npy(
        outlier_reconstruction,
        experiment_dir / "samples" / "outlier_reconstruction.npy",
    )

    # The paper uses 1,000 members from the training set and 1,000 non-members
    # from the MNIST test set. sample_mia_sets must preserve that distinction.
    mia_sets = sample_mia_sets(
        split,
        n_members=n_members,
        n_nonmembers=n_nonmembers,
        seed=seed,
    )

    member_images = mia_sets["member_images"]
    nonmember_images = mia_sets["nonmember_images"]

    member_distances = nearest_synthetic_distances(
        member_images, mia_synthetics
    )
    nonmember_distances = nearest_synthetic_distances(
        nonmember_images, mia_synthetics
    )

    curve = paper_tradeoff_curve(member_distances, nonmember_distances)

    # Distance-based membership score: smaller distance => more likely member.
    y_true = np.concatenate(
        [
            np.ones(len(member_distances), dtype=np.int32),
            np.zeros(len(nonmember_distances), dtype=np.int32),
        ]
    )
    y_score = -np.concatenate([member_distances, nonmember_distances])
    mia_auc = float(roc_auc_score(y_true, y_score))

    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        metrics_dir / "mia_tradeoff.csv",
        np.column_stack(
            [curve["threshold"], curve["type_i"], curve["type_ii"]]
        ),
        delimiter=",",
        header="threshold,type_i,type_ii",
        comments="",
    )
    np.save(metrics_dir / "member_nearest_distances.npy", member_distances)
    np.save(metrics_dir / "nonmember_nearest_distances.npy", nonmember_distances)

    save_mia_tradeoff_plot(
        curve["type_i"],
        curve["type_ii"],
        experiment_dir / "plots" / "mia_tradeoff.png",
        title="MIA trade-off (paper protocol; our result)",
    )

    metrics = {
        "paper": (
            "Groß & Wunder, Differentially Private Synthetic Data Generation "
            "via Lipschitz-Regularised Variational Autoencoders, Section V"
        ),
        "source": "OUR RESULT",
        "checkpoint": str(args.checkpoint),
        "seed": seed,
        "n_train_config": configured_n_train,
        "white_outlier_in_training_config": configured_outlier,
        "n_synthetic_for_mia": int(n_synthetic),
        "mia_synthetic_representation": args.mia_input,
        "n_members": int(len(member_distances)),
        "n_nonmembers": int(len(nonmember_distances)),
        "mia_roc_auc": mia_auc,
        "mean_member_nearest_distance": float(member_distances.mean()),
        "mean_nonmember_nearest_distance": float(nonmember_distances.mean()),
        "member_minus_nonmember_distance": float(
            member_distances.mean() - nonmember_distances.mean()
        ),
        "outlier_reconstruction_mean": float(
            outlier_reconstruction.mean().item()
        ),
        "outlier_reconstruction_min": float(
            outlier_reconstruction.min().item()
        ),
        "outlier_reconstruction_max": float(
            outlier_reconstruction.max().item()
        ),
        "note": (
            "The paper specifies 1000 members and 1000 non-members, but does "
            "not clearly state the number of synthetic records used for MIA. "
            "See the saved CSV and distance arrays for the full curve/results."
        ),
    }

    with open(metrics_dir / "vae_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation written to: {experiment_dir}")
    print(f"MIA ROC-AUC: {mia_auc:.4f}")
    print(f"Mean member distance: {member_distances.mean():.6f}")
    print(f"Mean non-member distance: {nonmember_distances.mean():.6f}")
    print(
        "Outlier reconstruction range: "
        f"[{outlier_reconstruction.min().item():.4f}, "
        f"{outlier_reconstruction.max().item():.4f}]"
    )


if __name__ == "__main__":
    main()
