import pytest

from mirl_ext.sft.modality_sampler import DistributedModalityHomogeneousSampler
from mirl_ext.sft.sft_dataset import _has_items


def test_has_items_handles_parquet_style_values():
    assert not _has_items(None)
    assert not _has_items([])
    assert _has_items(["frame.jpg"])


def test_distributed_sampler_shards_homogeneous_global_batches():
    flags = [False] * 18 + [True] * 17
    samplers = [
        DistributedModalityHomogeneousSampler(flags, 8, 2, rank, seed=5)
        for rank in range(2)
    ]
    rank_orders = [list(sampler) for sampler in samplers]

    assert len(rank_orders[0]) == len(rank_orders[1]) == 16
    seen = set()
    for start in range(0, len(rank_orders[0]), 4):
        global_batch = rank_orders[0][start : start + 4] + rank_orders[1][start : start + 4]
        assert len({flags[i] for i in global_batch}) == 1
        assert not (seen & set(global_batch))
        seen.update(global_batch)


def test_distributed_sampler_changes_order_by_epoch_in_lockstep():
    flags = [False] * 16 + [True] * 16
    samplers = [
        DistributedModalityHomogeneousSampler(flags, 8, 2, rank, seed=13)
        for rank in range(2)
    ]
    epoch_zero = [list(sampler) for sampler in samplers]
    for sampler in samplers:
        sampler.set_epoch(1)
    epoch_one = [list(sampler) for sampler in samplers]

    assert epoch_zero != epoch_one
    assert len(epoch_zero[0]) == len(epoch_one[0])


def test_distributed_sampler_drops_a_modality_short_of_one_batch():
    flags = [False] * 16 + [True] * 3
    sampler = DistributedModalityHomogeneousSampler(flags, 8, 2, 0)
    order = list(sampler)
    assert len(order) == len(sampler) == 8
    assert not any(flags[i] for i in order)


def test_distributed_sampler_rejects_when_nothing_fills_a_batch():
    with pytest.raises(ValueError, match="no media kind fills one global batch"):
        DistributedModalityHomogeneousSampler([False] * 5 + [True] * 3, 8, 2, 0)
