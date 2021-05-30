"""Limited Rank Matrix LVQ example using the Tecator dataset."""

import argparse

import pytorch_lightning as pl
import torch

import prototorch as pt

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Tecator(root="~/datasets/", train=True)
    test_ds = pt.datasets.Tecator(root="~/datasets/", train=False)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(
        distribution={
            "num_classes": 3,
            "prototypes_per_class": 4
        },
        input_dim=100,
        latent_dim=2,
        proto_lr=0.001,
        bb_lr=0.001,
    )

    # Initialize the model
    model = pt.models.GMLVQ(hparams,
                            prototype_initializer=pt.components.SMI(train_ds))

    # Callbacks
    vis = pt.models.VisSiameseGLVQ2D(train_ds, border=0.1)
    es = pl.callbacks.EarlyStopping(monitor="val_loss",
                                    min_delta=0.001,
                                    patience=3,
                                    verbose=False,
                                    mode="min")

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis, es],
    )

    # Training loop
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    torch.save(model, "liramlvq_tecator.pt")

    # Load a saved model
    saved_model = torch.load("liramlvq_tecator.pt")

    # Display the Lambda matrix
    saved_model.show_lambda()

    # Testing
    trainer.test(model, test_dataloaders=test_loader)
