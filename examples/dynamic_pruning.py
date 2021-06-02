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
        prune_after_epochs=30,  # pruning too early may cause problems
        prune_quota_per_epoch=1,  # prune at most 1 prototype per epoch
        frequency=5,  # prune every fifth epoch
        verbose=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        callbacks=[
            vis,
            pruning,
        ],
        terminate_on_nan=True,
        weights_summary=None,
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
