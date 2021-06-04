"""RSLVQ example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        distribution=[2, 2, 3],
        lr=0.05,
        variance=0.1,
    )

    # Initialize the model
    model = pt.models.probabilistic.RSLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        # prototype_initializer=pt.components.SMI(train_ds),
        prototype_initializer=pt.components.SSI(train_ds, noise=0.2),
        # prototype_initializer=pt.components.Zeros(2),
        # prototype_initializer=pt.components.Ones(2, scale=2.0),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        terminate_on_nan=True,
        weights_summary="full",
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
