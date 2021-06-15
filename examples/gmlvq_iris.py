"""GLVQ example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris()

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        input_dim=4,
        latent_dim=3,
        distribution={
            "num_classes": 3,
            "prototypes_per_class": 2
        },
        proto_lr=0.0005,
        bb_lr=0.0005,
    )

    # Initialize the model
    model = pt.models.GMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototype_initializer=pt.components.SSI(train_ds),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
        omega_initializer=pt.components.PCA(train_ds.data)
    )

    # Compute intermediate input and output sizes
    #model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = pt.models.VisGMLVQ2D(data=train_ds, border=0.1)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        weights_summary="full",
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
