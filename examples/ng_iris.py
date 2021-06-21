"""Neural Gas example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ExponentialLR

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Prepare and pre-process the dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=150)

    # Hyperparameters
    hparams = dict(
        num_prototypes=30,
        input_dim=2,
        lr=0.03,
    )

    # Initialize the model
    model = pt.models.NeuralGas(
        hparams,
        prototypes_initializer=pt.core.ZCI(2),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisNG2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        weights_summary="full",
    )

    # Training loop
    trainer.fit(model, train_loader)
