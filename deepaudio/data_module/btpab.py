from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing import Dict, List, Optional

import numpy as np

import torch
import torch.utils.data as data
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from .sampler import SegmentSampler
from .dataset import Dataset


def collate_fn(batch: List[Dict]) -> Dict:
    for waveform_dict in batch:
        for source_type in waveform_dict.keys():
            waveform_dict[source_type] = torch.from_numpy(
                waveform_dict[source_type]
            )
    return batch


class LitBTPAB(LightningDataModule):
    def __init__(
        self,
        sampler_keywords,
        dataset_keywords,
        num_workers,
        root: str=None
    ):
        r"""Data module.

        Args:
            dataset: Dataset object
            sampler: Sampler object
            valid_dataset: Dataset=None,
            valid_sampler: Sampler=None,
            test_dataset: Dataset=None,
            test_sampler: Sampler=None,
            num_workers: int
        """
        super().__init__()

        self.sampler = SegmentSampler(**sampler_keywords)
        self.dataset = Dataset(**dataset_keywords)
        self.num_workers = num_workers

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = data.DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.train_dataloader()
