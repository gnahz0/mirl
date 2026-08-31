"""MIRL SFT entry point with globally modality-homogeneous DP batches."""

from __future__ import annotations

import hydra
from torchdata.stateful_dataloader import StatefulDataLoader

from mirl_ext.sft.modality_sampler import DistributedModalityHomogeneousSampler
from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


class MIRLSFTTrainer(SFTTrainer):
    """The stock engine trainer with a synchronized image/video sampler."""

    def _make_loader(self, dataset, shuffle, label, seed, dp_size, dp_rank, device_name):
        sampler = DistributedModalityHomogeneousSampler(
            dataset.modality_flags,
            self.global_batch_size,
            dp_size,
            dp_rank,
            shuffle=shuffle,
            seed=seed,
        )
        if self.rank == 0:
            n_video = sum(dataset.modality_flags)
            print(
                f"MIRL homogeneous {label} sampler: "
                f"image={len(dataset) - n_video} video={n_video} "
                f"global_batch={self.global_batch_size} "
                f"steps={len(sampler) // self.train_batch_size_per_dp}"
            )
        loader = StatefulDataLoader(
            dataset=dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )
        return sampler, loader

    def _build_dataloader(self):
        config = self.config
        device_name = get_device_name()
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.global_batch_size = config.data.train_batch_size
        if self.global_batch_size % dp_size:
            raise ValueError("data.train_batch_size must be divisible by the data-parallel size")
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)
        seed = config.data.get("seed") or config.trainer.seed

        self.train_sampler, self.train_dataloader = self._make_loader(
            self.train_dataset, True, "train", seed, dp_size, dp_rank, device_name
        )
        if self.val_dataset:
            self.val_sampler, self.val_dataloader = self._make_loader(
                self.val_dataset, False, "val", seed, dp_size, dp_rank, device_name
            )
        else:
            self.val_dataloader = None


def run_sft(config) -> None:
    initialize_global_process_group()
    trainer = MIRLSFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="../../verl/trainer/config", config_name="sft_trainer_engine", version_base=None)
def main(config) -> None:
    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()
