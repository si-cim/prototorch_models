"""GMLVQ example using all four dimensions of the Iris dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import initializers as cinit
from prototorch.datasets.abstract import NumpyDataset
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from prototorch.models.callbacks.visualization import VisSiameseGLVQ2D
from prototorch.models.glvq import GMLVQ

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
        prototype_initializer=cinit.SMI(torch.Tensor(x_train),
                                        torch.Tensor(y_train)),
        input_dim=x_train.shape[1],
        latent_dim=2,
        lr=0.01,
    )

    # Initialize the model
    model = GMLVQ(hparams)

    # Model summary
    print(model)

    # Callbacks
    vis = VisSiameseGLVQ2D(x_train, y_train)

    # Namespace hook for the visualization to work
    model.backbone = model.omega_layer

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
