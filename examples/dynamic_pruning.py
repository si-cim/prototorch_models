"""Dynamically prune 'loser' prototypes in GLVQ-type models."""

import argparse
import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import (
    CELVQ,
    PruneLoserPrototypes,
    VisGLVQ2D,
)
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 4
    num_features = 2
    num_clusters = 1
    train_ds = pt.datasets.Random(
        num_samples=500,
        num_classes=num_classes,
        num_features=num_features,
        num_clusters=num_clusters,
        separation=3.0,
        seed=42,
    )

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    prototypes_per_class = num_clusters * 5
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        lr=0.2,
    )

    # Initialize the model
    model = CELVQ(
        hparams,
        prototypes_initializer=pt.initializers.FVCI(2, 3.0),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    logging.info(model)

    # Callbacks
    vis = VisGLVQ2D(train_ds)
    pruning = PruneLoserPrototypes(
        threshold=0.01,  # prune prototype if it wins less than 1%
        idle_epochs=20,  # pruning too early may cause problems
        prune_quota_per_epoch=2,  # prune at most 2 prototypes per epoch
        frequency=1,  # prune every epoch
        verbose=True,
    )
    es = EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=20,
        mode="min",
        verbose=True,
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
        detect_anomaly=True,
        log_every_n_steps=1,
        max_epochs=1000,
    )

    # Training loop
    trainer.fit(model, train_loader)
