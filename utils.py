from typing import Callable, List
from pathlib import Path
from torch import Tensor
import torch.nn as nn
import torch.utils.data as data
import shutil
import yaml
import librosa

import pytorch_lightning as pl


def read_yaml(config_yaml: str) -> dict:
    """Read config file to dictionary.

    Args:
        config_yaml: str

    Returns:
        configs: dict
    """
    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs


def get_data_module(config: dict) -> pl.LightningDataModule:
    if config['name'] == 'btpab':
        from deepaudio.data_module.btpab import LitBTPAB
        DataModule = LitBTPAB
    else:
        raise NotImplementedError
    return DataModule(**config['keywords'])


def get_model(config: dict) -> nn.Module:
    if config['name'] == 'scgad':
        from deepaudio.model.scgad import SCGAD
        Model = SCGAD
    return Model(**config['keywords'])


def get_loss_fn(config: dict) -> Callable[[Tensor, Tensor], Tensor]:
    from deepaudio.loss.module import L1_sp, L1_wav, L1_wav_L1_sp, SCL
    if config['name'] == 'l1_wav':
        Loss = L1_wav
    elif config['name'] == 'l1_sp':
        Loss = L1_sp
    elif config['name'] == 'l1_wav_l1_sp':
        Loss = L1_wav_L1_sp
    elif config['name'] == 'scl':
        Loss = SCL
    return Loss(**config['keywords'])