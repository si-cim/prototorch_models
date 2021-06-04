"""Localized-GMLVQ example using the Moons dataset."""

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
    train_ds = pt.datasets.Moons(num_samples=300, noise=0.2, seed=42)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=2)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=256,
                                               shuffle=True)

    # Hyperparameters
    hparams = dict(
        distribution=[1, 3],
        input_dim=2,
        latent_dim=2,
    )

    # Initialize the model
    model = pt.models.LGMLVQ(hparams,
                             prototype_initializer=pt.components.SMI(train_ds))

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)
    es = pl.callbacks.EarlyStopping(
        monitor="train_acc",
        min_delta=0.001,
        patience=20,
        mode="max",
        verbose=False,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            es,
        ],
        weights_summary="full",
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
