"""Growing Neural Gas example using the Iris dataset."""

import argparse
import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import GrowingNeuralGas, VisNG2D
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

    # Prepare the data
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_loader = DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        num_prototypes=5,
        input_dim=2,
        lr=0.1,
    )

    # Initialize the model
    model = GrowingNeuralGas(
        hparams,
        prototypes_initializer=pt.initializers.ZCI(2),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Model summary
    logging.info(model)

    # Callbacks
    vis = VisNG2D(data=train_loader)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
        ],
        max_epochs=100,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
