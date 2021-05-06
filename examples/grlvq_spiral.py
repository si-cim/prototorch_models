"""GMLVQ example using all four dimensions of the Iris dataset."""

import pytorch_lightning as pl
import torch
from prototorch.components import initializers as cinit
from prototorch.datasets.abstract import NumpyDataset
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from prototorch.models.callbacks.visualization import VisSiameseGLVQ2D
from prototorch.models.glvq import GRLVQ

from sklearn.preprocessing import StandardScaler

from prototorch.datasets.spiral import make_spiral


class PrintRelevanceCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module: GRLVQ):
        print(pl_module.relevance_profile)


if __name__ == "__main__":
    # Dataset
    x_train, y_train = make_spiral(n_samples=1000, noise=0.3)
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        nclasses=2,
        prototypes_per_class=20,
        prototype_initializer=cinit.SSI(torch.Tensor(x_train),
                                        torch.Tensor(y_train)),
        #prototype_initializer=cinit.UniformInitializer(2),
        input_dim=x_train.shape[1],
        lr=0.1,
        #transfer_function="sigmoid_beta",
    )

    # Initialize the model
    model = GRLVQ(hparams)

    # Model summary
    print(model)

    # Callbacks
    vis = VisSiameseGLVQ2D(x_train, y_train)
    debug = PrintRelevanceCallback()

    # Setup trainer
    trainer = pl.Trainer(max_epochs=200, callbacks=[vis, debug])

    # Training loop
    trainer.fit(model, train_loader)
