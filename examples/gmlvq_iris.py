"""GMLVQ example using all four dimensions of the Iris dataset."""

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
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)
    # Hyperparameters
    nclasses = 3
    prototypes_per_class = 1
    hparams = dict(
        distribution=(nclasses, prototypes_per_class),
        input_dim=x_train.shape[1],
        latent_dim=x_train.shape[1],
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = pt.models.GMLVQ(hparams,
                            prototype_initializer=pt.components.SMI(train_ds))

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(args, )

    # Training loop
    trainer.fit(model, train_loader)

    # Display the Lambda matrix
    model.show_lambda()
