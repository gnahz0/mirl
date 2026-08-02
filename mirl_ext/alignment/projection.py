# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Projection from encoder features into the shared contrastive space."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection, or an opt-in two-layer MLP when ``hidden_dim`` is set."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            if hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be positive or None, got {hidden_dim}")
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim, bias=bias),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCMSMLPEncoder(nn.Module):
    """SmellNet's released GC-MS encoder."""

    def __init__(self, in_dim: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCMSProjectionHead(nn.Module):
    """SmellNet's native 256-D encoder plus our shared-space adapter."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.encoder = GCMSMLPEncoder(in_dim)
        self.adapter = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encoder(x))
