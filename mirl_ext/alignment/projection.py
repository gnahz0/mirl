# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Small MLP projection head mapping encoder hidden -> shared contrastive dim."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """2-layer MLP with LayerNorm + GELU and optional dropout.

    ``forward`` returns *unnormalized* embeddings; callers should ``F.normalize(dim=-1)``
    before contrastive losses.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
