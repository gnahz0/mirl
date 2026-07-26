# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage 1 multimodal alignment: contrastive image/raw-signal/text + frozen-VE distillation.

Modules:
    projection:   small MLP projection heads to a shared embedding dim.
    losses:       symmetric InfoNCE + cosine distillation.
    data:         AlignmentDataset wrapping existing Parquet/JSONL rows (+ raw signals).
    model:        exact trainable/frozen Qwen3.5 vision towers + frozen SigLIP2
                  label-text tower + projections. Raw signals use merger-aware
                  pseudo-images or native temporal pseudo-video patches.
    trainer:      single-GPU PyTorch loop (no Ray, no veRL).

TODO(stage2): once Stage 1 weights are good, export `trainable_visual` back into a full
Qwen3.5 HF checkpoint and point veRL's `actor_rollout_ref.model.path` at it for SFT/RL.
"""
