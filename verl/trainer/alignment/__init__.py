# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage 1 multimodal alignment: contrastive image/time-series-image/text + frozen-VE distillation.

Modules:
    ts_renderer: render raw time series (CSV/ndarray) into PIL images for the VE.
    projection:  small MLP projection heads to a shared embedding dim.
    losses:      symmetric InfoNCE + MSE/KL distillation losses.
    data:        AlignmentDataset wrapping the repo's existing JSONL format.
    model:       MultimodalAlignmentModel = trainable Qwen3-VL VE + frozen ref + CLIP text + heads.
    trainer:     single-GPU PyTorch loop (no Ray, no veRL).

TODO(stage2): once Stage 1 weights are good, export `trainable_visual` back into a full
Qwen3-VL HF checkpoint (see `export.py`) and point veRL's
`actor_rollout_ref.model.path` at it for SFT/RL.
"""
