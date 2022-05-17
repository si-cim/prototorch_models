"""GMLVQ example using the spiral dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import (
    GMLVQ,
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
    train_ds = pt.datasets.Spiral(num_samples=500, noise=0.5)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    num_classes = 2
    prototypes_per_class = 10
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        transfer_function="swish_beta",
        transfer_beta=10.0,
        proto_lr=0.1,
        bb_lr=0.1,
        input_dim=2,
        latent_dim=2,
    )

    # Initialize the model
    model = GMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SSCI(train_ds, noise=1e-2),
    )

    # Callbacks
    vis = VisGLVQ2D(
        train_ds,
        show_last_only=False,
        block=False,
    )
    pruning = PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=10,
        prune_quota_per_epoch=5,
        frequency=5,
        replace=True,
        prototypes_initializer=pt.initializers.SSCI(train_ds, noise=1e-1),
        verbose=True,
    )
    es = EarlyStopping(
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
            es,
            pruning,
        ],
        max_epochs=1000,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
