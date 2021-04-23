"""CBC example using the Iris dataset."""

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.cbc import CBC


class VisualizationCallback(pl.Callback):
    def __init__(self,
                 x_train,
                 y_train,
                 title="Prototype Visualization",
                 cmap="viridis"):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.title = title
        self.fig = plt.figure(self.title)
        self.cmap = cmap

    def on_epoch_end(self, trainer, pl_module):
        # protos = pl_module.prototypes
        protos = pl_module.components
        # plabels = pl_module.prototype_labels
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel("Data dimension 1")
        ax.set_ylabel("Data dimension 2")
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
        ax.scatter(
            protos[:, 0],
            protos[:, 1],
            # c=plabels,
            c="k",
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )
        x = np.vstack((x_train, protos))
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / 50),
                             np.arange(y_min, y_max, 1 / 50))
        mesh_input = np.c_[xx.ravel(), yy.ravel()]
        y_pred = pl_module.predict(torch.Tensor(mesh_input))
        y_pred = y_pred.reshape(xx.shape)

        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)
        ax.set_xlim(left=x_min + 0, right=x_max - 0)
        ax.set_ylim(bottom=y_min + 0, top=y_max - 0)
        plt.pause(0.1)


if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        input_dim=x_train.shape[1],
        nclasses=3,
        prototypes_per_class=3,
        prototype_initializer="stratified_mean",
        lr=0.01,
    )

    # Initialize the model
    model = CBC(hparams, data=[x_train, y_train])

    # Fix the component locations
    # model.proto_layer.requires_grad_(False)

    # Pure-positive reasonings
    ncomps = 3
    nclasses = 3
    rmat = torch.stack(
        [0.9 * torch.eye(ncomps),
         torch.zeros(ncomps, nclasses)], dim=0)
    # model.reasoning_layer.load_state_dict({"reasoning_probabilities": rmat},
    #                                       strict=True)

    print(model.reasoning_layer.reasoning_probabilities)
    # import sys
    # sys.exit()

    # Model summary
    print(model)

    # Callbacks
    vis = VisualizationCallback(x_train, y_train)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            vis,
        ],
    )

    # Training loop
    trainer.fit(model, train_loader)
