#!/usr/bin/env python3
"""Generate samples and run Monte Carlo MIA from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.mnist import build_mnist_paper_data, sample_mia_sets
from src.evaluation.generate import generate_samples, reconstruction_grid, save_samples_npy
from src.evaluation.mia import monte_carlo_nearest_attack
from src.evaluation.plots import save_image_grid, save_mia_tradeoff_plot
from src.training.trainer import load_checkpoint
from src.utils.device import get_device
from src.utils.experiment import create_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved VAE checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, default="eval_run")
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, config = load_checkpoint(args.checkpoint, device=device)
    experiment_dir = create_experiment_dir(config["experiment"]["results_root"], args.experiment_id)

    n_synth = args.n_synthetic or int(config["eval"]["n_synthetic"])
    n_members = int(config["eval"]["n_members"])
    n_nonmembers = int(config["eval"]["n_nonmembers"])
    if args.smoke:
        n_synth, n_members, n_nonmembers = 64, 16, 16

    split = build_mnist_paper_data(
        data_dir=config["dataset"]["data_dir"],
        n_train=int(config["dataset"]["n_train"]),
        binarize_threshold=float(config["dataset"]["binarize_threshold"]),
        add_white_outlier=bool(config["dataset"]["add_white_outlier"]),
        seed=int(config["train"]["seed"]),
    )
    samples = generate_samples(model, n=n_synth, device=device, postprocess=config["eval"]["sample_postprocess"])
    save_samples_npy(samples, experiment_dir / "samples" / "generated_samples.npy")
    save_image_grid(samples[: int(config["eval"]["n_sample_grid"])], experiment_dir / "plots" / "generated_samples.png")

    outlier = split.outlier.unsqueeze(0)
    grid = reconstruction_grid(model, outlier, device)
    save_image_grid(grid, experiment_dir / "plots" / "outlier_reconstruction.png", nrow=2, title="white outlier (real | recon)")

    mia_sets = sample_mia_sets(split, n_members, n_nonmembers, seed=int(config["train"]["seed"]))
    attack = monte_carlo_nearest_attack(
        mia_sets["member_images"].numpy(),
        mia_sets["nonmember_images"].numpy(),
        samples.numpy(),
    )
    np.savetxt(
        experiment_dir / "metrics" / "mia_tradeoff.csv",
        np.column_stack([attack["threshold"], attack["type_i"], attack["type_ii"]]),
        delimiter=",",
        header="threshold,type_i,type_ii",
        comments="",
    )
    save_mia_tradeoff_plot(attack["type_i"], attack["type_ii"], experiment_dir / "plots" / "mia_tradeoff.png")
    with open(experiment_dir / "metrics" / "vae_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "OUR RESULT",
                "n_synthetic": n_synth,
                "n_members": n_members,
                "n_nonmembers": n_nonmembers,
                "mean_member_distance": float(attack["member_scores"].mean()),
                "mean_nonmember_distance": float(attack["nonmember_scores"].mean()),
            },
            handle,
            indent=2,
        )
    print(f"Wrote evaluation to {experiment_dir}")


if __name__ == "__main__":
    main()
