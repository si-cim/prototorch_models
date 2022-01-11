"""LVQMLN example using all four dimensions of the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch


class Backbone(torch.nn.Module):

    def __init__(self, input_size=4, hidden_size=10, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dense1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.dense1(x))
        out = self.activation(self.dense2(x))
        return out


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=150)

    # Hyperparameters
    hparams = dict(
        distribution=[3, 4, 5],
        proto_lr=0.001,
        bb_lr=0.001,
    )

    # Initialize the backbone
    backbone = Backbone()

    # Initialize the model
    model = pt.models.LVQMLN(
        hparams,
        prototypes_initializer=pt.initializers.SSCI(
            train_ds,
            transform=backbone,
        ),
        backbone=backbone,
    )

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisSiameseGLVQ2D(
        data=train_ds,
        map_protos=False,
        border=0.1,
        resolution=500,
        axis_off=True,
    )
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=20,
        prune_quota_per_epoch=2,
        frequency=10,
        verbose=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            pruning,
        ],
    )

    # Training loop
    trainer.fit(model, train_loader)
