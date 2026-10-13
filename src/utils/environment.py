"""Record software versions for a run."""

from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import torch

from src.utils.device import get_device


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except Exception:
        info["sklearn"] = None
    return info
