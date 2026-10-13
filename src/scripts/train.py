#!/usr/bin/env python3
"""Train one paper VAE variant. Outputs go under results/<experiment_id>/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.mnist import build_mnist_paper_data, make_loader
from src.training.trainer import build_model, train_vae
from src.utils.config import load_config, merge_config
from src.utils.device import get_device
from src.utils.environment import collect_environment
from src.utils.experiment import create_experiment_dir
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper VAE variant")
    parser.add_argument("--lipschitz-L", type=float)
    parser.add_argument("--gp-lambda", type=float)
    parser.add_argument("--noise-multiplier", type=float)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--variant", choices=["vanilla", "lvae_gp", "lvae_sn", "dpsgd"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--smoke", action="store_true", help="Tiny subset / few epochs for local debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_path = ROOT / "configs" / "default.yaml"
    config = load_config(default_path)
    if args.config.resolve() != default_path.resolve():
        config = merge_config(config, load_config(args.config))
    override: dict = {}
    if args.variant:
        override["train"] = {"variant": args.variant}
        if args.variant == "lvae_gp":
            override["lvae"] = {"method": "gradient_penalty"}
        elif args.variant == "lvae_sn":
            override["lvae"] = {"method": "spectral_norm"}

    if args.lipschitz_L is not None:
        override.setdefault("lvae", {})["lipschitz_L"] = args.lipschitz_L
    if args.gp_lambda is not None:
        override.setdefault("lvae", {})["gp_lambda"] = args.gp_lambda
    if args.noise_multiplier is not None:
        override.setdefault("dpsgd", {})["noise_multiplier"] = args.noise_multiplier
    if args.epochs is not None:
        override.setdefault("train", {})["epochs"] = args.epochs
    if args.n_train is not None:
        override.setdefault("dataset", {})["n_train"] = args.n_train
    if args.experiment_id:
        override.setdefault("experiment", {})["id"] = args.experiment_id
    if args.smoke:
        override = merge_config(
            override,
            {
                "dataset": {"n_train": 128},
                "train": {"epochs": 2, "batch_size": 32},
                "experiment": {"id": "smoke_vanilla_mnist"},
            },
        )
    config = merge_config(config, override)

    set_seed(int(config["train"]["seed"]))
    device = get_device()
    experiment_dir = create_experiment_dir(config["experiment"]["results_root"], config["experiment"]["id"])
    with open(experiment_dir / "logs" / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2)

    split = build_mnist_paper_data(
        data_dir=config["dataset"]["data_dir"],
        n_train=int(config["dataset"]["n_train"]),
        binarize_threshold=float(config["dataset"]["binarize_threshold"]),
        add_white_outlier=bool(config["dataset"]["add_white_outlier"]),
        seed=int(config["train"]["seed"]),
    )
    np.save(experiment_dir / "logs" / "train_indices.npy", split.train_indices)
    loader = make_loader(
        split.train_images,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    model = build_model(config)
    train_vae(model, loader, config, experiment_dir, device=device)
    print(f"Wrote experiment to {experiment_dir}")


if __name__ == "__main__":
    main()
