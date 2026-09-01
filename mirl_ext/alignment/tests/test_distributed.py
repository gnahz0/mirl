
from __future__ import annotations

import math
import os
from contextlib import nullcontext
from unittest import mock

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
        self.bias = torch.nn.Parameter(torch.tensor(-10.0))

    def forward(self, x):
        values = torch.nn.functional.normalize(x @ self.weight, dim=-1)
        return values, self.log_scale * 1.0, self.bias * 1.0


def _free_port() -> int:
    """Claim an unused port for one rendezvous: a hardcoded port makes concurrent
    runs deadlock on the rendezvous, orphaning gloo processes on a shared login node."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_metrics(z, labels, candidates, text, world_size=1):
    """Score one exclusive family through the unified path."""
    from mirl_ext.alignment.metrics import (
        _bank_metrics,
        _bank_stats,
        all_reduce_sum,
        build_bank_specs,
    )

    spec = build_bank_specs({"ecg": (candidates, text)}, None)[0]
    ids = torch.tensor([candidates.index(label) for label in labels])
    target = torch.nn.functional.one_hot(ids, num_classes=len(candidates))
    stats = all_reduce_sum(_bank_stats(z, target, spec), world_size=world_size)
    return _bank_metrics(stats, spec)


def _worker(rank: int, results: dict, port: int) -> None:
    from mirl_ext.alignment.data import TACTILE_SPANS
    from mirl_ext.alignment.objective import _label_siglip_loss, _tactile_siglip_loss

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        inputs = (torch.tensor([[1.0, 0.0], [0.8, 0.2]]), torch.tensor([[0.0, 1.0]]))[rank]
        labels = (["A", "A"], ["B"])[rank]
        candidates = ("A", "B")
        text = torch.eye(2)

        model = DistributedDataParallel(_ToyModel())
        for sync in (False, True):
            with nullcontext() if sync else model.no_sync():
                embeddings, scale, bias = model(inputs)
                loss = _label_siglip_loss(
                    embeddings,
                    labels,
                    candidates,
                    text,
                    scale,
                    bias,
                    world_size=2,
                )
                (loss / 2).backward()
        metrics = _rank_metrics(embeddings.detach(), labels, candidates, text, world_size=2)

        reference = _ToyModel()
        all_inputs = torch.cat((torch.tensor([[1.0, 0.0], [0.8, 0.2]]), torch.tensor([[0.0, 1.0]])))
        all_embeddings, all_scale, all_bias = reference(all_inputs)
        reference_loss = _label_siglip_loss(
            all_embeddings,
            ["A", "A", "B"],
            candidates,
            text,
            all_scale,
            all_bias,
        )
        reference_loss.backward()
        reference_metrics = _rank_metrics(all_embeddings.detach(), ["A", "A", "B"], candidates, text)
        standard_result = (
            torch.allclose(model.module.weight.grad, reference.weight.grad, atol=1e-6),
            torch.allclose(model.module.log_scale.grad, reference.log_scale.grad, atol=1e-6),
            torch.allclose(model.module.bias.grad, reference.bias.grad, atol=1e-6),
            metrics == pytest.approx(reference_metrics),
        )

        model.zero_grad()
        reference.zero_grad()
        embeddings, scale, bias = model(inputs)
        tactile_targets = (torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))[rank]
        # Rank 1's only row is unannotated, so the sharded loss must equal a reference over rank 0's rows alone.
        tactile_mask = (torch.ones(2, 2), torch.zeros(1, 2))[rank]
        # The toy bank is 2 labels wide; pretend it is one task spanning both.
        with mock.patch.dict(TACTILE_SPANS, {"toy": (0, 2)}, clear=True):
            tactile_loss, _ = _tactile_siglip_loss(
                embeddings,
                tactile_targets,
                tactile_mask,
                text,
                scale,
                bias,
                world_size=2,
            )
            assert tactile_loss is not None
            tactile_loss.backward()

            reference_embeddings, reference_scale, reference_bias = reference(all_inputs[:2])
            reference_loss, _ = _tactile_siglip_loss(
                reference_embeddings,
                torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                torch.ones(2, 2),
                text,
                reference_scale,
                reference_bias,
            )
            assert reference_loss is not None
            reference_loss.backward()
        results[rank] = standard_result + (
            torch.allclose(model.module.weight.grad, reference.weight.grad, atol=1e-6),
            torch.allclose(model.module.log_scale.grad, reference.log_scale.grad, atol=1e-6),
            torch.allclose(model.module.bias.grad, reference.bias.grad, atol=1e-6),
        )
    finally:
        dist.destroy_process_group()


def test_ddp_local_loss_and_metrics_match_the_global_batch():
    ctx = mp.get_context("spawn")
    results = ctx.Manager().dict()
    port = _free_port()
    processes = [ctx.Process(target=_worker, args=(rank, results, port)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert all(all(result) for result in results.values())


def _asymmetric_worker(rank: int, results: dict, port: int) -> None:
    """Each rank holds rows for a family the other rank has none of."""
    from mirl_ext.alignment.metrics import (
        all_reduce_sum,
        build_bank_specs,
        new_stats,
        prediction_metrics,
        update_stats,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        candidates = ("A", "B")
        bank = {"ecg": (candidates, torch.eye(2)), "tactile": (candidates, torch.eye(2))}
        specs = build_bank_specs(bank, None)
        smell_z = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        ecg_z = torch.tensor([[0.0, 1.0], [0.9, 0.1], [0.3, 0.7]])

        # Rank 0 sees only ecg, rank 1 only tactile -- the shape the old per-family reduce could not agree on.
        shard = (
            (smell_z, ["A", "B"], ["ecg"] * 2),
            (ecg_z, ["B", "A", "B"], ["tactile"] * 3),
        )[rank]
        stats = new_stats(specs, torch.device("cpu"))
        update_stats(stats, specs, (shard[0], shard[1], shard[2], None, None), None, None)
        sharded = prediction_metrics(all_reduce_sum(stats, world_size=2), specs)

        reference_stats = new_stats(specs, torch.device("cpu"))
        update_stats(
            reference_stats,
            specs,
            (
                torch.cat([smell_z, ecg_z]),
                ["A", "B", "B", "A", "B"],
                ["ecg"] * 2 + ["tactile"] * 3,
                None,
                None,
            ),
            None,
            None,
        )
        reference = prediction_metrics(reference_stats, specs)
        results[rank] = (set(sharded) == set(reference), sharded == pytest.approx(reference))
    finally:
        dist.destroy_process_group()


def test_metrics_agree_when_a_rank_holds_no_rows_for_a_family():
    """A rank with zero rows of a family must still reach the same collective:
    the reduce buffer is sized from the label banks pre-data, so it contributes
    exact zeros rather than skipping the reduce (which would hang, not fail)."""
    ctx = mp.get_context("spawn")
    results = ctx.Manager().dict()
    port = _free_port()
    processes = [ctx.Process(target=_asymmetric_worker, args=(rank, results, port)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert len(results) == 2
    assert all(all(result) for result in results.values())
