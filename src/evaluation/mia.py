"""Monte Carlo-style membership inference (Section II-C, V-A).

Paper: threshold the distance of a target to its nearest synthetic records.
ASSUMED metric: Euclidean distance on flattened images.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances


def _as_matrix(images: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(images, dtype=np.float64)
    return array.reshape(array.shape[0], -1)


def nearest_synthetic_distances(targets: np.ndarray, synthetics: np.ndarray) -> np.ndarray:
    target_mat = _as_matrix(targets)
    synth_mat = _as_matrix(synthetics)
    distances = pairwise_distances(target_mat, synth_mat, metric="euclidean")
    return distances.min(axis=1)


def tradeoff_curve(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    n_thresholds: int = 200,
) -> dict[str, np.ndarray]:
    """Predict member if distance < τ. Type I = false alarm on non-members."""
    all_scores = np.concatenate([member_scores, nonmember_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)
    type_i = []
    type_ii = []
    for tau in thresholds:
        false_alarm = np.mean(nonmember_scores < tau)
        miss = np.mean(member_scores >= tau)
        type_i.append(false_alarm)
        type_ii.append(miss)
    return {
        "threshold": thresholds,
        "type_i": np.asarray(type_i),
        "type_ii": np.asarray(type_ii),
    }


def monte_carlo_nearest_attack(
    member_images: np.ndarray,
    nonmember_images: np.ndarray,
    synthetic_images: np.ndarray,
    n_thresholds: int = 200,
) -> dict[str, Any]:
    member_scores = nearest_synthetic_distances(member_images, synthetic_images)
    nonmember_scores = nearest_synthetic_distances(nonmember_images, synthetic_images)
    curve = tradeoff_curve(member_scores, nonmember_scores, n_thresholds=n_thresholds)
    return {
        "member_scores": member_scores,
        "nonmember_scores": nonmember_scores,
        **curve,
    }
