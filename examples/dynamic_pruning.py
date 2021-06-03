"""Dynamically prune 'loser' prototypes in GLVQ-type models."""

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
    num_classes = 4
    num_features = 2
    num_clusters = 1
    train_ds = pt.datasets.Random(num_samples=500,
                                  num_classes=num_classes,
                                  num_features=num_features,
                                  num_clusters=num_clusters,
                                  separation=3.0,
                                  seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    prototypes_per_class = num_clusters * 5
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        lr=0.3,
    )

    # Initialize the model
    model = pt.models.CELVQ(
        hparams,
        prototype_initializer=pt.components.Ones(2, scale=3),
    )

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds)
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,  # prune prototype if it wins less than 1%
        idle_epochs=10,  # pruning too early may cause problems
        prune_quota_per_epoch=5,  # prune at most 5 prototypes per epoch
        frequency=2,  # prune every second epoch
        verbose=True,
    )

    es = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=15,
        mode="min",
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            pruning,
            es,
        ],
        terminate_on_nan=True,
        weights_summary=None,
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
