"""Siamese GLVQ example using all four dimensions of the Iris dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import initializers as cinit
from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.callbacks.visualization import VisSiameseGLVQ2D
from prototorch.models.glvq import SiameseGLVQ
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader


class Backbone(torch.nn.Module):
    """Two fully connected layers with ReLU activation."""
    def __init__(self, input_size=4, hidden_size=10, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dense1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        out = self.relu(self.dense2(x))
        return out


if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    train_ds = NumpyDataset(x_train, y_train)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=2)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        nclasses=3,
        prototypes_per_class=2,
        prototype_initializer=cinit.SMI(torch.Tensor(x_train),
                                        torch.Tensor(y_train)),
        proto_lr=0.001,
        bb_lr=0.001,
    )

    # Initialize the model
    model = SiameseGLVQ(
        hparams,
        backbone_module=Backbone,
    )

    # Model summary
    print(model)

    # Callbacks
    vis = VisSiameseGLVQ2D(x_train, y_train, border=0.1)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
