"""Warm-starting GLVQ with prototypes from Growing Neural Gas."""

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

    # Prepare the data
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Initialize the gng
    gng = pt.models.GrowingNeuralGas(
        hparams=dict(num_prototypes=5, insert_freq=2, lr=0.1),
        prototypes_initializer=pt.initializers.ZCI(2),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Callbacks
    es = pl.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=20,
        mode="min",
        verbose=False,
        check_on_train_epoch_end=True,
    )

    # Setup trainer for GNG
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[es],
        weights_summary=None,
    )

    # Training loop
    trainer.fit(gng, train_loader)

    # Hyperparameters
    hparams = dict(
        distribution=[],
        lr=0.01,
    )

    # Warm-start prototypes
    knn = pt.models.KNN(dict(k=1), data=train_ds)
    prototypes = gng.prototypes
    plabels = knn.predict(prototypes)

    # Initialize the model
    model = pt.models.GLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.LCI(prototypes),
        labels_initializer=pt.initializers.LLI(plabels),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.02,
        idle_epochs=2,
        prune_quota_per_epoch=5,
        frequency=1,
        verbose=True,
    )
    es = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=10,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            pruning,
            es,
        ],
        weights_summary="full",
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
