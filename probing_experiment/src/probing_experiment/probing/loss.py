from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """Append a homogeneous 1 coordinate to (..., N, 2) point tensors."""
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with final dim 2, got {points.shape}")

    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def sampson_error(
    fundamental: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    First-order geometric reprojection error for corresponding image points.

    Args:
        fundamental:
            Fundamental matrices with shape (..., 3, 3)
        pts1:
            Pixel coordinates from image 1 with shape (..., N, 2).
        pts2:
            Pixel coordinates from image 2 with shape (..., N, 2).
        eps:
            Small value preventing division by zero for degenerate inputs.

    Returns:
        Per-correspondence Sampson errors with shape broadcast from (..., N).
    """
    if fundamental.shape[-2:] != (3, 3):
        raise ValueError(f"Expected F with shape (..., 3, 3), got {fundamental.shape}")

    pts1_h = to_homogeneous(pts1)
    pts2_h = to_homogeneous(pts2)

    f_x1 = torch.matmul(fundamental, pts1_h.unsqueeze(-1)).squeeze(-1)
    ft_x2 = torch.matmul(fundamental.transpose(-1, -2), pts2_h.unsqueeze(-1)).squeeze(-1)
    x2t_f_x1 = (pts2_h * f_x1).sum(dim=-1)

    denom = (
        f_x1[..., 0].square()
        + f_x1[..., 1].square()
        + ft_x2[..., 0].square()
        + ft_x2[..., 1].square()
    ).clamp_min(eps)

    return x2t_f_x1.square() / denom


def mean_sampson_error(
    fundamental: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean Sampson error over all correspondences and batch dimensions."""
    return sampson_error(fundamental, pts1, pts2, eps=eps).mean()


def scale_fundamental_matrix(
    fundamental: torch.Tensor,
    scale1: tuple[float, float],
    scale2: tuple[float, float],
) -> torch.Tensor:
    """
    Transform F when image coordinates are scaled.

    If resized coordinates are x1' = T1 x1 and x2' = T2 x2, the resized-image
    fundamental matrix is F' = T2^{-T} F T1^{-1}.
    """
    dtype = fundamental.dtype
    device = fundamental.device

    sx1, sy1 = scale1
    sx2, sy2 = scale2
    t1_inv = torch.tensor(
        [[1.0 / sx1, 0.0, 0.0], [0.0, 1.0 / sy1, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    t2_inv = torch.tensor(
        [[1.0 / sx2, 0.0, 0.0], [0.0, 1.0 / sy2, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )

    return t2_inv.transpose(-1, -2) @ fundamental @ t1_inv
