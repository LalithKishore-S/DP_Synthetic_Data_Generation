"""Experiment output directories. Never silently overwrite existing results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


RESULT_SUBDIRS = ("models", "metrics", "plots", "samples", "logs")


def _is_occupied(path: Path) -> bool:
    if not path.exists():
        return False
    for child in path.rglob("*"):
        if child.is_file() and child.name != ".gitkeep":
            return True
    return False


def create_experiment_dir(results_root: str | Path, experiment_id: str) -> Path:
    root = Path(results_root)
    candidate = root / experiment_id
    if _is_occupied(candidate):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / f"{experiment_id}_{stamp}"
    for sub in RESULT_SUBDIRS:
        (candidate / sub).mkdir(parents=True, exist_ok=True)
    return candidate
