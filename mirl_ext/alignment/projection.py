# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Projection from encoder features into the shared contrastive space."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear shared-space adapter."""

    def __init__(self, in_dim: int, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
