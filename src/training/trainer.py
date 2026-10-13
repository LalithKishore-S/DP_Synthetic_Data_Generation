"""Training loop for vanilla / L-VAE / DP-SGD variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses import decoder_gradient_penalty, elbo_loss
from src.models.vae import VAE
from src.utils.device import get_device
from src.utils.environment import collect_environment
import time
from src.evaluation.plots import save_training_plot


def build_model(config: dict[str, Any]) -> VAE:
    model_cfg = config["model"]
    lvae_cfg = config.get("lvae", {})
    variant = config["train"].get("variant", "vanilla")
    use_sn = variant == "lvae_sn" or lvae_cfg.get("method") == "spectral_norm"
    return VAE(
        latent_dim=int(model_cfg["latent_dim"]),
        conv_filters=tuple(model_cfg["conv_filters"]),
        encoder_hidden=int(model_cfg["encoder_hidden"]),
        decoder_hidden=int(model_cfg["decoder_hidden"]),
        decoder_reshape=tuple(model_cfg["decoder_reshape"]),
        tconv_filters=tuple(model_cfg["tconv_filters"]),
        spectral_norm=use_sn,
        lipschitz_L=float(lvae_cfg.get("lipschitz_L", 1.0)),
    )


def _maybe_attach_opacus(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    config: dict[str, Any],
) -> tuple[VAE, torch.optim.Optimizer, DataLoader, Any]:
    if config["train"].get("variant") != "dpsgd":
        return model, optimizer, loader, None
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise ImportError(
            "DP-SGD variant requires the optional 'opacus' package. "
            "Install it or run vanilla / L-VAE."
        ) from exc

    dpsgd = config["dpsgd"]
    engine = PrivacyEngine()
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=float(dpsgd["noise_multiplier"]),
        max_grad_norm=float(dpsgd["max_grad_norm"]),
    )
    return model, optimizer, loader, engine


def train_vae(
    model: VAE,
    loader: DataLoader,
    config: dict[str, Any],
    experiment_dir: Path,
    device: torch.device | None = None,
    progress: bool = True,
    step_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    device = device or get_device()
    model = model.to(device)
    train_cfg = config["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    model, optimizer, loader, privacy_engine = _maybe_attach_opacus(model, optimizer, loader, config)

    variant = train_cfg.get("variant", "vanilla")
    beta = float(config["loss"].get("beta_kl", 1.0))
    gp_lambda = float(config.get("lvae", {}).get("gp_lambda", 0.0))
    lipschitz_L = float(config.get("lvae", {}).get("lipschitz_L", 1.0))
    use_gp = variant == "lvae_gp" or config.get("lvae", {}).get("method") == "gradient_penalty"

    history: list[dict[str, float]] = []
    epochs = int(train_cfg["epochs"])
    epoch_iter = range(epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="epochs")

    model.train()
    for epoch in epoch_iter:
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "gp": 0.0, "n": 0.0}
        for batch in loader:
            x = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            terms = elbo_loss(out["logits"], x, out["mu"], out["logvar"], beta_kl=beta)
            loss = terms["loss"]
            gp_value = torch.zeros((), device=device)
            if use_gp:
                # gp_value = decoder_gradient_penalty(model.decoder, out["z"], lipschitz_L)
                z_probe = torch.cat([out["z"].detach(), torch.randn_like(out["z"])], dim=0)
                gp_value = decoder_gradient_penalty(
                    model.decoder, z_probe, lipschitz_L,
                    n_projections=int(config.get("lvae", {}).get("gp_n_projections", 1)),
                )
                loss = loss + gp_lambda * gp_value
            loss.backward()
            optimizer.step()

            batch_n = x.shape[0]
            running["loss"] += float(loss.detach()) * batch_n
            running["recon"] += float(terms["recon"].detach()) * batch_n
            running["kl"] += float(terms["kl"].detach()) * batch_n
            running["gp"] += float(gp_value.detach()) * batch_n
            running["n"] += batch_n

        n = max(running["n"], 1.0)
        row = {
            "epoch": float(epoch + 1),
            "loss": running["loss"] / n,
            "recon": running["recon"] / n,
            "kl": running["kl"] / n,
            "gp": running["gp"] / n,
        }
        if privacy_engine is not None:
            delta = float(config["dpsgd"]["delta"])
            row["epsilon"] = float(privacy_engine.get_epsilon(delta))
        history.append(row)
        if step_callback:
            step_callback(row)

    _write_training_artifacts(experiment_dir, model, config, history, privacy_engine)
    save_training_plot(history, experiment_dir / "plots" / "training_loss.png")
    return {"history": history, "model": model, "privacy_engine": privacy_engine}


def _write_training_artifacts(
    experiment_dir: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    history: list[dict[str, float]],
    privacy_engine: Any,
) -> None:
    models_dir = experiment_dir / "models"
    logs_dir = experiment_dir / "logs"
    metrics_dir = experiment_dir / "metrics"
    checkpoint = {
        "model_state": _unwrap_module(model).state_dict(),
        "config": config,
        "environment": collect_environment(),
    }
    torch.save(checkpoint, models_dir / "vae_final.pt")
    torch.save(checkpoint, models_dir / "vae_best.pt")

    with open(metrics_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)

    payload = {
        "config": config,
        "environment": collect_environment(),
        "n_epochs": len(history),
        "dpsgd_epsilon": history[-1].get("epsilon") if history else None,
        "privacy_engine": type(privacy_engine).__name__ if privacy_engine is not None else None,
    }
    with open(logs_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_module", model)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[VAE, dict[str, Any]]:
    device = device or get_device()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, config
