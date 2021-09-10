"""k-NN example using the Iris dataset from scikit-learn."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    X, y = load_iris(return_X_y=True)
    X = X[:, [0, 2]]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.5,
                                                        random_state=42)

    train_ds = pt.datasets.NumpyDataset(X_train, y_train)
    test_ds = pt.datasets.NumpyDataset(X_test, y_test)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)

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
        data=(X_train, y_train),
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
    y_pred = model.predict(torch.tensor(X_train))
    print(y_pred)

    # Test
    trainer.test(model, dataloaders=test_loader)
