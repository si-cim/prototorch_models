"""CBC example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch

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
        distribution=[1, 0, 3],
        margin=0.1,
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = pt.models.CBC(
        hparams,
        components_initializer=pt.initializers.SSCI(train_ds, noise=0.01),
        reasonings_iniitializer=pt.initializers.
        PurePositiveReasoningsInitializer(),
    )

    # Callbacks
    vis = pt.models.Visualize2DVoronoiCallback(
        data=train_ds,
        title="CBC Iris Example",
        resolution=100,
        axis_off=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)
