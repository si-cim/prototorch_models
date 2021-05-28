"""Probabilistic-LVQ example using the Iris dataset."""

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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    num_classes = 3
    prototypes_per_class = 2
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        lr=0.05,
        variance=1.0,
    )

    # Initialize the model
    model = pt.models.probabilistic.LikelihoodRatioLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        # prototype_initializer=pt.components.UniformInitializer(2),
        prototype_initializer=pt.components.SMI(train_ds),
    )

    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)
