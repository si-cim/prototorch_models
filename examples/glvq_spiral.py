"""GLVQ example using the spiral dataset."""

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
    train_ds = pt.datasets.Spiral(num_samples=500, noise=0.5)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    num_classes = 2
    prototypes_per_class = 10
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        transfer_function="swish_beta",
        transfer_beta=10.0,
        # lr=0.1,
        proto_lr=0.1,
        bb_lr=0.1,
        input_dim=2,
        latent_dim=2,
    )

    # Initialize the model
    model = pt.models.GMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototype_initializer=pt.components.SSI(train_ds, noise=1e-2),
    )

    # Callbacks
    vis = pt.models.VisGLVQ2D(
        train_ds,
        show_last_only=False,
        block=False,
    )
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.02,
        idle_epochs=10,
        prune_quota_per_epoch=5,
        frequency=2,
        replace=True,
        initializer=pt.components.SSI(train_ds, noise=1e-2),
        verbose=True,
    )
    es = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=1.0,
        patience=5,
        mode="min",
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            # es,
            pruning,
        ],
        terminate_on_nan=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
