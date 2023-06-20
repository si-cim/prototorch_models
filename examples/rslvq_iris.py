"""RSLVQ example using the Iris dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import RSLVQ, VisGLVQ2D
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
    seed_everything(seed=42)

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        distribution=[2, 2, 3],
        proto_lr=0.05,
        lambd=0.1,
        variance=1.0,
        input_dim=2,
        latent_dim=2,
        bb_lr=0.01,
    )

    # Initialize the model
    model = RSLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SSCI(train_ds, noise=0.2),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
        ],
        detect_anomaly=True,
        max_epochs=100,
        log_every_n_steps=1,
    )

    # Training loop
    trainer.fit(model, train_loader)
