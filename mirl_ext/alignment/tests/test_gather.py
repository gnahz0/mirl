# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

SHARED_DIM = 8


def _run(rank: int, world: int, body_name: str, out: dict, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        globals()[body_name](rank, world, out)
    finally:
        dist.destroy_process_group()


def _spawn(body_name: str, port: str, world: int = 3) -> dict:
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    processes = [ctx.Process(target=_run, args=(rank, world, body_name, out, port)) for rank in range(world)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    for process in processes:
        if process.is_alive():
            process.terminate()
            pytest.fail(f"{body_name}: distributed gather hung")
        assert process.exitcode == 0
    return dict(out)


def _body_uneven(rank: int, world: int, out: dict) -> None:
    from mirl_ext.alignment.objective import _gather_ts_embeddings

    metadata_group = dist.new_group(backend="gloo")
    counts = [2, 0, 1]
    families_by_rank = ["smell", "ecg", "tactile"]
    n = counts[rank]
    z = torch.full((n, SHARED_DIM), float(rank + 1), requires_grad=True) if n else None
    labels = [f"r{rank}_{i}" for i in range(n)]
    families = [families_by_rank[rank]] * n

    gathered, gathered_labels, gathered_families = _gather_ts_embeddings(
        z,
        labels,
        families,
        torch.device("cpu"),
        world,
        shared_dim=SHARED_DIM,
        metadata_group=metadata_group,
    )
    gathered.sum().backward()
    out[f"rows_{rank}"] = [
        (label, family, float(row[0]))
        for label, family, row in zip(gathered_labels, gathered_families, gathered, strict=True)
    ]
    out[f"grad_{rank}"] = float(z.grad.norm()) if z is not None else None
    out[f"backward_{rank}"] = True


def test_gather_handles_uneven_rows_metadata_and_backward():
    out = _spawn("_body_uneven", port="29517")
    assert out["rows_0"] == out["rows_1"] == out["rows_2"]
    assert len(out["rows_0"]) == 3
    for label, family, value in out["rows_0"]:
        owner = int(label[1])
        assert value == pytest.approx(owner + 1)
        assert family == ("smell" if owner == 0 else "tactile")
    assert out["grad_0"] > 0 and out["grad_2"] > 0
    assert all(out[f"backward_{rank}"] for rank in range(3))
