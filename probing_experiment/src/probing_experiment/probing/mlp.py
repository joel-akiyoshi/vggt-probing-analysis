from __future__ import annotations

import torch
from torch import nn


def normalize_fundamental_frobenius(
    fundamental: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize fundamental matrices by Frobenius norm."""
    if fundamental.shape[-2:] != (3, 3):
        raise ValueError(f"Expected F with shape (..., 3, 3), got {fundamental.shape}")

    norm = torch.linalg.matrix_norm(fundamental, ord="fro", dim=(-2, -1))
    return fundamental / norm.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)


class FundamentalMatrixMLP(nn.Module):
    """One-hidden-layer probe from VGGT camera-token features to F."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9),
        )

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized fundamental matrix predictions."""
        return self.net(x).reshape(-1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Frobenius-normalized fundamental matrix predictions."""
        return normalize_fundamental_frobenius(self.forward_raw(x), eps=self.eps)
