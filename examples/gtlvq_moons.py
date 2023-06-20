"""Localized-GTLVQ example using the Moons dataset."""

import argparse
import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import GTLVQ, VisGLVQ2D
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Reproducibility
    seed_everything(seed=2)

    # Dataset
    train_ds = pt.datasets.Moons(num_samples=300, noise=0.2, seed=42)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
    )

    # Hyperparameters
    # Latent_dim should be lower than input dim.
    hparams = dict(distribution=[1, 3], input_dim=2, latent_dim=1)

    # Initialize the model
    model = GTLVQ(hparams,
                  prototypes_initializer=pt.initializers.SMCI(train_ds))

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    logging.info(model)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)
    es = EarlyStopping(
        monitor="train_acc",
        min_delta=0.001,
        patience=20,
        mode="max",
        verbose=False,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
            es,
        ],
        max_epochs=1000,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
