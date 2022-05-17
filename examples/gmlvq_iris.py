"""GMLVQ example using the Iris dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import GMLVQ, VisGMLVQ2D
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.optim.lr_scheduler import ExponentialLR
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
    train_ds = pt.datasets.Iris()

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        input_dim=4,
        latent_dim=4,
        distribution={
            "num_classes": 3,
            "per_class": 2
        },
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = GMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 4)

    # Callbacks
    vis = VisGMLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
        ],
        max_epochs=100,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
