"""Neural Gas example using the Iris dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import NeuralGas, VisNG2D
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
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

    # Prepare and pre-process the dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, 0:3:2]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=150)

    # Hyperparameters
    hparams = dict(
        num_prototypes=30,
        input_dim=2,
        lr=0.03,
    )

    # Initialize the model
    model = NeuralGas(
        hparams,
        prototypes_initializer=pt.core.ZCI(2),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = VisNG2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
        ],
        max_epochs=1000,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
