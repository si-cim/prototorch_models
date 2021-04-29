"""Siamese GLVQ example using all four dimensions of the Iris dataset."""

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.glvq import SiameseGLVQ


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
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        x_train = pl_module.backbone(torch.Tensor(x_train)).detach()
        protos = pl_module.backbone(torch.Tensor(protos)).detach()
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.axis("off")
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
        ax.scatter(
            protos[:, 0],
            protos[:, 1],
            c=plabels,
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )
        x = np.vstack((x_train, protos))
        x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
        y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / 50),
                             np.arange(y_min, y_max, 1 / 50))
        mesh_input = np.c_[xx.ravel(), yy.ravel()]
        y_pred = pl_module.predict_latent(torch.Tensor(mesh_input))
        y_pred = y_pred.reshape(xx.shape)

        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)
        ax.set_xlim(left=x_min + 0, right=x_max - 0)
        ax.set_ylim(bottom=y_min + 0, top=y_max - 0)
        tb = pl_module.logger.experiment
        tb.add_figure(
            tag=f"{self.title}",
            figure=self.fig,
            global_step=trainer.current_epoch,
            close=False,
        )
        plt.pause(0.1)


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
        input_dim=x_train.shape[1],
        nclasses=3,
        prototypes_per_class=1,
        prototype_initializer="stratified_mean",
        lr=0.01,
    )

    # Initialize the model
    model = SiameseGLVQ(hparams,
                        backbone_module=Backbone,
                        data=[x_train, y_train])

    # Model summary
    print(model)

    # Callbacks
    vis = VisualizationCallback(x_train, y_train)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
