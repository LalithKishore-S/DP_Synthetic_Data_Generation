from __future__ import annotations

from pathlib import Path

import torch

from src.data.mnist import make_loader
from src.training.trainer import build_model, train_vae
from src.utils.config import load_config
from src.utils.experiment import create_experiment_dir


def test_small_data_training_runs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    config["train"]["epochs"] = 2
    config["train"]["batch_size"] = 16
    images = (torch.rand(48, 1, 28, 28) >= 0.5).float()
    loader = make_loader(images, batch_size=16, shuffle=True)
    experiment_dir = create_experiment_dir(tmp_path, "smoke_train")
    model = build_model(config)
    result = train_vae(model, loader, config, experiment_dir, device=torch.device("cpu"), progress=False)
    history = result["history"]
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    # Learning is not guaranteed in 2 epochs on noise; only require a finite logged loss.
    assert history[0]["recon"] > 0
