"""Median-LVQ example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=len(train_ds),  # MedianLVQ cannot handle mini-batches
    )

    # Initialize the model
    model = pt.models.MedianLVQ(
        hparams=dict(distribution=(3, 2), lr=0.01),
        prototypes_initializer=pt.initializers.SSCI(train_ds),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)
    es = pl.callbacks.EarlyStopping(
        monitor="train_acc",
        min_delta=0.01,
        patience=5,
        mode="max",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis, es],
        weights_summary="full",
    )

    # Training loop
    trainer.fit(model, train_loader)
