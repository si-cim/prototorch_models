"""k-NN example using the Iris dataset from scikit-learn."""

import argparse
import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models import KNN, VisGLVQ2D
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Dataset
    X, y = load_iris(return_X_y=True)
    X = X[:, 0:3:2]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
    )

    train_ds = pt.datasets.NumpyDataset(X_train, y_train)
    test_ds = pt.datasets.NumpyDataset(X_test, y_test)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Hyperparameters
    hparams = dict(k=5)

    # Initialize the model
    model = KNN(hparams, data=train_ds)

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    logging.info(model)

    # Callbacks
    vis = VisGLVQ2D(
        data=(X_train, y_train),
        resolution=200,
        block=True,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        max_epochs=1,
        callbacks=[
            vis,
        ],
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    # This is only for visualization. k-NN has no training phase.
    trainer.fit(model, train_loader)

    # Recall
    y_pred = model.predict(torch.tensor(X_train))
    logging.info(y_pred)

    # Test
    trainer.test(model, dataloaders=test_loader)
