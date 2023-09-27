import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from utils import (
    read_yaml,
    get_data_module,
    get_model,
    get_loss_fn
)


def train(args) -> None:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """
    # arugments & parameters
    workspace = Path(args.workspace)

    # Read config file.
    configs: dict = read_yaml(args.config_yaml)

    # data_module
    data_module = get_data_module(configs['data_module'])

    # model
    model = get_model(configs['model'])

    loss_fn = get_loss_fn(configs['loss_fn'])

    # pytorch-lightning model
    from deepaudio.pl_module.binaural_rendering import LitBinauralRendering
    pl_model = LitBinauralRendering(
        model=model,
        loss_fn=loss_fn,
    )

    if 'checkpoint_path' in configs['lightning_module']:
        checkpoint_path = configs['lightning_module']['checkpoint_path']
        pl_model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            loss_fn=loss_fn
        )
        print(
            "Load pl.LightningModule from {}".format(checkpoint_path)
        )

    logger = TensorBoardLogger(
        save_dir=workspace/'logs',
        name=configs['experiment']['name'],
        version=configs['experiment']['version'],
    )

    # save checkpoint callback
    checkpoints_dir = workspace/"checkpoints"/Path(args.config_yaml).stem

    save_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="epoch",
        filename="{epoch:03d}",
        mode="max",
        save_top_k=3,
        every_n_epochs=2,
        auto_insert_metric_name=False
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        logger=logger,
        callbacks=[save_checkpoint_callback, lr_monitor_callback],
        **configs["trainer"]["keywords"]
    )

    trainer.fit(pl_model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    # parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    args = parser.parse_args()

    train(args)
