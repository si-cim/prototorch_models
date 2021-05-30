"""Dynamically update the number of prototypes in GLVQ."""

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

    for _ in range(5):
        # Callbacks
        vis = pt.models.VisGLVQ2D(train_ds)

        # Setup trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            max_epochs=20,
            callbacks=[vis],
            terminate_on_nan=True,
            weights_summary=None,
        )

        # Training loop
        trainer.fit(model, train_loader)

        # Increase prototypes
        model.increase_prototypes(
            pt.components.SMI(train_ds),
            distribution=[1, 1, 1],
        )
