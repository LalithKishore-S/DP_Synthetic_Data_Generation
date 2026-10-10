#!/usr/bin/env python3
"""Local smoke path: random binary images, 2 epochs, checkpoint, generate."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.mnist import make_loader
from src.evaluation.generate import generate_samples
from src.training.trainer import build_model, load_checkpoint, train_vae
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.experiment import create_experiment_dir
from src.utils.reproducibility import set_seed


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["train"]["epochs"] = 2
    config["train"]["batch_size"] = 32
    config["train"]["variant"] = "vanilla"
    config["experiment"]["id"] = "smoke_random_binary"
    set_seed(0)
    device = get_device()
    images = (torch.rand(128, 1, 28, 28) >= 0.5).float()
    loader = make_loader(images, batch_size=32, shuffle=True)
    experiment_dir = create_experiment_dir(ROOT / "results", config["experiment"]["id"])
    model = build_model(config)
    train_vae(model, loader, config, experiment_dir, device=device, progress=False)
    loaded, _ = load_checkpoint(experiment_dir / "models" / "vae_final.pt", device=device)
    samples = generate_samples(loaded, n=8, device=device)
    assert samples.shape == (8, 1, 28, 28)
    print(f"SMOKE_OK device={device} dir={experiment_dir} sample_mean={float(samples.mean()):.4f}")


if __name__ == "__main__":
    main()
