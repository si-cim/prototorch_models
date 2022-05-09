"""GLVQ example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        distribution={
            "num_classes": 3,
            "per_class": 4
        },
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        weights_summary="full",
    )

    # Training loop
    trainer.fit(model, train_loader)

    # Manual save
    trainer.save_checkpoint("./glvq_iris.ckpt")

    # Load saved model
    new_model = pt.models.GLVQ.load_from_checkpoint(
        checkpoint_path="./glvq_iris.ckpt",
        strict=False,
    )
    print(new_model)
