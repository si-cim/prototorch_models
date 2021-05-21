"""GLVQ example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)

    # Hyperparameters
    nclasses = 3
    prototypes_per_class = 2
    hparams = dict(
        distribution=(nclasses, prototypes_per_class),
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(hparams,
                           optimizer=torch.optim.Adam,
                           prototype_initializer=pt.components.SMI(train_ds))

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=(x_train, y_train), block=False)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)
