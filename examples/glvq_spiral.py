"""GLVQ example using the spiral dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import initializers as cinit
from prototorch.datasets.abstract import NumpyDataset
from prototorch.datasets.spiral import make_spiral
from prototorch.models.callbacks.visualization import VisGLVQ2D
from prototorch.models.glvq import GLVQ
from torch.utils.data import DataLoader


class StopOnNaN(pl.Callback):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def on_epoch_end(self, trainer, pl_module, logs={}):
        if torch.isnan(self.param).any():
            raise ValueError("NaN encountered. Stopping.")


if __name__ == "__main__":
    # Dataset
    x_train, y_train = make_spiral(n_samples=600, noise=0.6)
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=256)

    # Hyperparameters
    hparams = dict(
        nclasses=2,
        prototypes_per_class=20,
        # prototype_initializer=cinit.SSI(torch.Tensor(x_train),
        prototype_initializer=cinit.SMI(torch.Tensor(x_train),
                                        torch.Tensor(y_train)),
        lr=0.01,
    )

    # Initialize the model
    model = GLVQ(hparams)

    # Callbacks
    vis = VisGLVQ2D(x_train, y_train)
    # vis = VisGLVQ2D(x_train, y_train, show_last_only=True, block=True)
    snan = StopOnNaN(model.proto_layer.components)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[vis, snan],
    )

    # Training loop
    trainer.fit(model, train_loader)
