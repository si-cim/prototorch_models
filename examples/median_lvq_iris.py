"""Median-LVQ example using the Iris dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import MedianLVQ, VisGLVQ2D
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
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=len(train_ds),  # MedianLVQ cannot handle mini-batches
    )

    # Initialize the model
    model = MedianLVQ(
        hparams=dict(distribution=(3, 2), lr=0.01),
        prototypes_initializer=pt.initializers.SSCI(train_ds),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)
    es = EarlyStopping(
        monitor="train_acc",
        min_delta=0.01,
        patience=5,
        mode="max",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
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
