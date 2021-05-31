"""Dynamically update the number of prototypes in GLVQ."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class PrototypeScheduler(Callback):
    def __init__(self, train_ds, freq=20):
        self.train_ds = train_ds
        self.freq = freq

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.freq == 0:
            pl_module.increase_prototypes(
                pt.components.SMI(self.train_ds),
                distribution=[1, 1, 1],
            )


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(
        distribution=[1, 1, 1],
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(
        hparams,
        prototype_initializer=pt.components.SMI(train_ds),
    )

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds)
    proto_scheduler = PrototypeScheduler(train_ds, 10)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        callbacks=[
            vis,
            proto_scheduler,
        ],
        terminate_on_nan=True,
        weights_summary=None,
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
