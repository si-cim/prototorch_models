"""k-NN example using the Iris dataset from scikit-learn."""

import argparse

import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris

import prototorch as pt

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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=150)

    # Hyperparameters
    hparams = dict(k=5)

    # Initialize the model
    model = pt.models.KNN(hparams, data=train_ds)

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(
        data=(x_train, y_train),
        resolution=200,
        block=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1,
        callbacks=[vis],
        weights_summary="full",
    )

    # Training loop
    # This is only for visualization. k-NN has no training phase.
    trainer.fit(model, train_loader)

    # Recall
    y_pred = model.predict(torch.tensor(x_train))
    print(y_pred)
