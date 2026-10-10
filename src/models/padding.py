"""TensorFlow-style SAME padding for stride-2 convolutions.

The paper implements the VAE in TensorFlow 2.6 (Section V-A) and does not
state PyTorch padding. This matches TF `padding='same'` spatial sizes:
28 → 14 → 7 → 4 → 2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pad_same(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    _, _, height, width = x.shape
    out_h = math.ceil(height / stride)
    out_w = math.ceil(width / stride)
    pad_h = max((out_h - 1) * stride + kernel_size - height, 0)
    pad_w = max((out_w - 1) * stride + kernel_size - width, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
