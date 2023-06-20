"""Siamese GLVQ example using all four dimensions of the Iris dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import SiameseGLVQ, VisSiameseGLVQ2D
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Backbone(torch.nn.Module):

    def __init__(self, input_size=4, hidden_size=10, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dense1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.dense1(x))
        out = self.activation(self.dense2(x))
        return out


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris()

    # Reproducibility
    seed_everything(seed=2)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=150)

    # Hyperparameters
    hparams = dict(
        distribution=[1, 2, 3],
        lr=0.01,
    )

    # Initialize the backbone
    backbone = Backbone()

    # Initialize the model
    model = SiameseGLVQ(
        hparams,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
        backbone=backbone,
        both_path_gradients=False,
    )

    # Callbacks
    vis = VisSiameseGLVQ2D(data=train_ds, border=0.1)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
        ],
        max_epochs=1000,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
