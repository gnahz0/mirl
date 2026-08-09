# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from mirl_ext.alignment.runtime import load_checkpoint, load_training_state, save_checkpoint


class _TinyAlignmentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable_visual = torch.nn.Linear(2, 2)
        self.log_logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))


def test_checkpoint_restores_model_optimizer_scheduler_and_progress(tmp_path):
    model = _TinyAlignmentModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))

    (model.trainable_visual(torch.ones(1, 2)).sum() + model.log_logit_scale).backward()
    optimizer.step()
    scheduler.step()
    expected_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_lr = optimizer.param_groups[0]["lr"]

    save_checkpoint(
        model,
        tmp_path,
        OmegaConf.create({"test": True}),
        step=7,
        optimizer=optimizer,
        scheduler=scheduler,
        progress={
            "best_value": 0.5,
            "next_epoch": 0,
            "next_batch_index": 28,
            "steps_per_epoch": 10,
            "total_steps": 20,
        },
    )

    restored = _TinyAlignmentModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda step: 1.0 / (step + 1),
    )

    assert load_checkpoint(restored, tmp_path) == 7
    progress = load_training_state(tmp_path, restored_optimizer, restored_scheduler)

    for key, expected in expected_weights.items():
        torch.testing.assert_close(restored.state_dict()[key], expected)
    assert restored_optimizer.state_dict()["state"]
    assert restored_optimizer.param_groups[0]["lr"] == expected_lr
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert progress == {
        "step": 7,
        "best_value": 0.5,
        "next_epoch": 0,
        "next_batch_index": 28,
        "steps_per_epoch": 10,
        "total_steps": 20,
    }
