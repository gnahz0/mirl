"""Distributed sampler whose synchronized global batches use one media kind."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class DistributedModalityHomogeneousSampler(Sampler[int]):
    """Build image/video global batches first, then shard each across DP ranks.

    A vanilla ``DistributedSampler`` shuffles and shards rows independently of
    modality. One rank can therefore enter Qwen's image path while its peer
    enters the video path, which desynchronizes FSDP collectives. Here every
    rank receives its local slice of the same homogeneous global batch.
    """

    def __init__(
        self,
        flags,
        global_batch_size: int,
        num_replicas: int,
        rank: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if global_batch_size <= 0 or global_batch_size % num_replicas:
            raise ValueError("global_batch_size must be positive and divisible by num_replicas")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.flags = [bool(flag) for flag in flags]
        self.global_batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Rows beyond the last full global batch of their kind are dropped each
        # epoch (< one global batch per kind; keeps small val splits usable).
        if self.flags and not len(self):
            raise ValueError(f"no media kind fills one global batch of {global_batch_size}")

    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _blocks(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = (
            torch.randperm(len(self.flags), generator=generator).tolist()
            if self.shuffle
            else list(range(len(self.flags)))
        )
        groups = ([i for i in permutation if not self.flags[i]], [i for i in permutation if self.flags[i]])
        per_kind = [
            [group[i : i + self.global_batch_size] for i in range(0, len(group), self.global_batch_size)]
            for group in groups
        ]
        per_kind = [[block for block in blocks if len(block) == self.global_batch_size] for blocks in per_kind]
        if self.shuffle:
            for blocks in per_kind:
                order = torch.randperm(len(blocks), generator=generator).tolist()
                blocks[:] = [blocks[i] for i in order]

        # Alternate while both modalities remain so even short smoke runs cover
        # both model paths; exhaust the larger modality afterward.
        blocks = []
        first = int(torch.randint(0, 2, (1,), generator=generator).item()) if self.shuffle else 0
        positions = [0, 0]
        while positions[0] < len(per_kind[0]) or positions[1] < len(per_kind[1]):
            for kind in (first, 1 - first):
                if positions[kind] < len(per_kind[kind]):
                    blocks.append(per_kind[kind][positions[kind]])
                    positions[kind] += 1
        return blocks

    def __iter__(self):
        start = self.rank * self.local_batch_size
        stop = start + self.local_batch_size
        return iter([index for block in self._blocks() for index in block[start:stop]])

    def __len__(self) -> int:
        used = sum(self.flags) // self.global_batch_size
        used += (len(self.flags) - sum(self.flags)) // self.global_batch_size
        return used * self.local_batch_size
