"""GLVQ example using the spiral dataset."""

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
    train_ds = pt.datasets.Spiral(num_samples=600, noise=0.6)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)

    # Hyperparameters
    num_classes = 2
    prototypes_per_class = 20
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(hparams,
                           prototype_initializer=pt.components.SSI(train_ds,
                                                                   noise=1e-1))

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=True, block=True)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        terminate_on_nan=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
