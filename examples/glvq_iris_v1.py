"""GLVQ example using the Iris dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import initializers as cinit
from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.callbacks.visualization import VisGLVQ2D
from prototorch.models.glvq import GLVQ
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        nclasses=3,
        prototypes_per_class=2,
        prototype_initializer=cinit.StratifiedMeanInitializer(
            torch.Tensor(x_train), torch.Tensor(y_train)),
        lr=0.01,
    )

    # Initialize the model
    model = GLVQ(hparams)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[VisGLVQ2D(x_train, y_train)],
    )

    # Training loop
    trainer.fit(model, train_loader)
