from __future__ import annotations

from pathlib import Path

import torch

from src.data.mnist import make_loader
from src.training.trainer import build_model, load_checkpoint, train_vae
from src.utils.config import load_config
from src.utils.experiment import create_experiment_dir


def test_checkpoint_save_load(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 8
    config["experiment"]["id"] = "test_ckpt"
    images = (torch.rand(16, 1, 28, 28) >= 0.5).float()
    loader = make_loader(images, batch_size=8, shuffle=False)
    experiment_dir = create_experiment_dir(tmp_path, "test_ckpt")
    model = build_model(config)
    train_vae(model, loader, config, experiment_dir, device=torch.device("cpu"), progress=False)

    ckpt = experiment_dir / "models" / "vae_final.pt"
    assert ckpt.exists()
    loaded, loaded_cfg = load_checkpoint(ckpt, device=torch.device("cpu"))
    orig = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new = {k: v.detach().cpu() for k, v in loaded.state_dict().items()}
    assert orig.keys() == new.keys()
    for key in orig:
        assert torch.allclose(orig[key], new[key])
    x = images[:2]
    with torch.no_grad():
        torch.manual_seed(0)
        a = model.to("cpu")(x)["logits"]
        torch.manual_seed(0)
        b = loaded(x)["logits"]
    assert torch.allclose(a, b, atol=1e-5)
    assert loaded_cfg["model"]["latent_dim"] == config["model"]["latent_dim"]
