"""Siamese GLVQ example using all four dimensions of the Iris dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import (
    StratifiedMeanInitializer
)
from prototorch.datasets.abstract import NumpyDataset
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from prototorch.models.callbacks.visualization import VisSiameseGLVQ2D
from prototorch.models.glvq import SiameseGLVQ


class Backbone(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=10, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dense1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.dense2(self.relu(self.dense1(x))))


if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        nclasses=3,
        prototypes_per_class=1,
        prototype_initializer=StratifiedMeanInitializer(
            torch.Tensor(x_train), torch.Tensor(y_train)),
        lr=0.01,
    )

    # Initialize the model
    model = SiameseGLVQ(
        hparams,
        backbone_module=Backbone,
    )

    # Model summary
    print(model)

    # Callbacks
    vis = VisSiameseGLVQ2D(x_train, y_train)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
