# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
import os
from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, x):
        values = torch.nn.functional.normalize(x @ self.weight, dim=-1)
        return values, self.log_scale * 1.0


def _worker(rank: int, results: dict) -> None:
    from mirl_ext.alignment.metrics import _label_ranking_metrics
    from mirl_ext.alignment.objective import _label_siglip_loss, _tactile_task_siglip_loss

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        inputs = (torch.tensor([[1.0, 0.0], [0.8, 0.2]]), torch.tensor([[0.0, 1.0]]))[rank]
        labels = (["A", "A"], ["B"])[rank]
        candidates = ("A", "B")
        text = torch.eye(2)

        model = DistributedDataParallel(_ToyModel())
        for sync in (False, True):
            with nullcontext() if sync else model.no_sync():
                embeddings, scale = model(inputs)
                loss = _label_siglip_loss(embeddings, labels, candidates, text, scale, world_size=2)
                (loss / 2).backward()
        metrics = _label_ranking_metrics(embeddings.detach(), labels, candidates, text, world_size=2)

        reference = _ToyModel()
        all_inputs = torch.cat((torch.tensor([[1.0, 0.0], [0.8, 0.2]]), torch.tensor([[0.0, 1.0]])))
        all_embeddings, all_scale = reference(all_inputs)
        reference_loss = _label_siglip_loss(all_embeddings, ["A", "A", "B"], candidates, text, all_scale)
        reference_loss.backward()
        reference_metrics = _label_ranking_metrics(all_embeddings.detach(), ["A", "A", "B"], candidates, text)
        standard_result = (
            torch.allclose(model.module.weight.grad, reference.weight.grad, atol=1e-6),
            torch.allclose(model.module.log_scale.grad, reference.log_scale.grad, atol=1e-6),
            metrics == pytest.approx(reference_metrics),
        )

        model.zero_grad()
        reference.zero_grad()
        embeddings, scale = model(inputs)
        tactile_targets = (torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))[rank]
        tactile_mask = (torch.tensor([True, True]), torch.tensor([False]))[rank]
        tactile_loss = _tactile_task_siglip_loss(
            embeddings,
            tactile_targets,
            tactile_mask,
            text,
            0.0,
            scale,
            world_size=2,
        )
        assert tactile_loss is not None
        tactile_loss.backward()

        reference_embeddings, reference_scale = reference(all_inputs[:2])
        reference_loss = _tactile_task_siglip_loss(
            reference_embeddings,
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([True, True]),
            text,
            0.0,
            reference_scale,
        )
        assert reference_loss is not None
        reference_loss.backward()
        results[rank] = standard_result + (
            torch.allclose(model.module.weight.grad, reference.weight.grad, atol=1e-6),
            torch.allclose(model.module.log_scale.grad, reference.log_scale.grad, atol=1e-6),
        )
    finally:
        dist.destroy_process_group()


def test_ddp_local_loss_and_metrics_match_the_global_batch():
    ctx = mp.get_context("spawn")
    results = ctx.Manager().dict()
    processes = [ctx.Process(target=_worker, args=(rank, results)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert all(all(result) for result in results.values())
