from typing import Any, Callable, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from .utils import (
    calculate_sdr,
    calculate_dild_in_db, calculate_ditd_in_step,
    WarmupAndReduce
)


class LitBinauralRendering(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            model: nn.Module
            loss: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn

    def training_step(self, batch: Dict, _) -> Dict:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch: e.g. [{
                "ambisonic": (batch_size, channels_num, segment_samples),
                "binaural": (batch_size, channels_num, segment_samples),
            }]

        Returns:
            loss: float, loss function of this mini-batch
        """
        x = batch['ambisonic']
        target = batch['binaural']

        # Forward.
        y = self.model(x)

        loss = self.loss_fn(
            input=y,
            target=target,
        )

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: List[Dict], _) -> None:
        # print(batch)
        x = batch['ambisonic']
        target = batch['binaural']

        y = self.model(x)

        loss = self.loss_fn(
            input=y,
            target=target,
        )

        sdr = calculate_sdr(y, target)
        ditd = calculate_ditd_in_step(y, target)
        dild = calculate_dild_in_db(y, target)
        # print(dild)
        self.log_dict({
            "val/loss": loss,
            "val/sdr": sdr,
            "val/ditd": ditd,
            "val/dild": dild
        })

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[Dict]]:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )
        scheduler = WarmupAndReduce(
            optimizer=optimizer,
            warmup_steps=1000,
            reduce_steps=10000,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
