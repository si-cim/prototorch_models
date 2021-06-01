"""CBC example using the Iris dataset."""

import argparse

import pytorch_lightning as pl
import torch

import prototorch as pt

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(
        distribution=[2, 2, 2],
        proto_lr=0.1,
    )

    # Initialize the model
    model = pt.models.CBC(
        hparams,
        prototype_initializer=pt.components.SSI(train_ds, noise=0.01),
    )

    # Callbacks
    vis = pt.models.VisCBC2D(data=train_ds,
                             title="CBC Iris Example",
                             resolution=100,
                             axis_off=True)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)
