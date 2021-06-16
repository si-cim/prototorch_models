"""Limited Rank Matrix LVQ example using the Tecator dataset."""

import argparse

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

import prototorch as pt


def plot_matrix(matrix):
    title = "Lambda matrix"
    plt.figure(title)
    plt.title(title)
    plt.imshow(matrix, cmap="gray")
    plt.axis("off")
    plt.colorbar()
    plt.show(block=True)


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Tecator(root="~/datasets/", train=True)
    test_ds = pt.datasets.Tecator(root="~/datasets/", train=False)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=10)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(
        distribution={
            "num_classes": 2,
            "prototypes_per_class": 1
        },
        input_dim=100,
        latent_dim=2,
        proto_lr=0.0001,
        bb_lr=0.0001,
    )

    # Initialize the model
    model = pt.models.SiameseGMLVQ(
        hparams,
        # optimizer=torch.optim.SGD,
        optimizer=torch.optim.Adam,
        prototype_initializer=pt.components.SMI(train_ds),
    )

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisSiameseGLVQ2D(train_ds, border=0.1)
    es = pl.callbacks.EarlyStopping(monitor="val_loss",
                                    min_delta=0.001,
                                    patience=50,
                                    verbose=False,
                                    mode="min")

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis, es],
        weights_summary=None,
    )

    # Training loop
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    torch.save(model, "liramlvq_tecator.pt")

    # Load a saved model
    saved_model = torch.load("liramlvq_tecator.pt")

    # Display the Lambda matrix
    plot_matrix(saved_model.lambda_matrix)

    # Testing
    trainer.test(model, test_dataloaders=test_loader)
