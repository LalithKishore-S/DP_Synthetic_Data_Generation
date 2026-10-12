from __future__ import annotations
from pathlib import Path
import torch
from src.data.mnist import make_loader
from src.training.trainer import build_model, train_vae
from src.utils.config import load_config
from src.utils.experiment import create_experiment_dir


def test_model_learns_on_structured_data(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    config["train"].update(epochs=100, batch_size=32)
    torch.manual_seed(0)
    protos = (torch.rand(2, 1, 28, 28) >= 0.5).float()
    images = protos.repeat(32, 1, 1, 1)
    loader = make_loader(images, batch_size=32, shuffle=True)
    d = create_experiment_dir(tmp_path, "learn")
    res = train_vae(build_model(config), loader, config, d, device=torch.device("cpu"), progress=False)
    h = res["history"]
    assert h[-1]["loss"] < 0.85 * h[0]["loss"]