"""CBC example using the Iris dataset."""

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.cbc import CBC


class VisualizationCallback(pl.Callback):
    def __init__(
        self,
        x_train,
        y_train,
        prototype_model=True,
        title="Prototype Visualization",
        cmap="viridis",
    ):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.title = title
        self.fig = plt.figure(self.title)
        self.cmap = cmap
        self.prototype_model = prototype_model

    def on_epoch_end(self, trainer, pl_module):
        if self.prototype_model:
            protos = pl_module.prototypes
            color = pl_module.prototype_labels
        else:
            protos = pl_module.components
            color = "k"
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel("Data dimension 1")
        ax.set_ylabel("Data dimension 2")
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
        ax.scatter(
            protos[:, 0],
            protos[:, 1],
            c=color,
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


def make_spirals(n_samples=500, noise=0.3):
    def get_samples(n, delta_t):
        points = []
        for i in range(n):
            r = i / n_samples * 5
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) + np.random.rand(1) * noise
            y = r * np.cos(t) + np.random.rand(1) * noise
            points.append([x, y])
        return points

    n = n_samples // 2
    positive = get_samples(n=n, delta_t=0)
    negative = get_samples(n=n, delta_t=np.pi)
    x = np.concatenate(
        [np.array(positive).reshape(n, -1),
         np.array(negative).reshape(n, -1)],
        axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return x, y


if __name__ == "__main__":
    # Dataset
    x_train, y_train = make_spirals(n_samples=1000, noise=0.3)
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        input_dim=x_train.shape[1],
        nclasses=2,
        prototypes_per_class=40,
        prototype_initializer="stratified_random",
        lr=0.05,
    )

    # Initialize the model
    model_class = CBC
    model = model_class(hparams, data=[x_train, y_train])

    # Pure-positive reasonings
    new_reasoning = torch.zeros_like(
        model.reasoning_layer.reasoning_probabilities)
    for i, label in enumerate(model.proto_layer.prototype_labels):
        new_reasoning[0][0][i][int(label)] = 1.0

    model.reasoning_layer.reasoning_probabilities.data = new_reasoning

    # Model summary
    print(model)

    # Callbacks
    vis = VisualizationCallback(x_train,
                                y_train,
                                prototype_model=hasattr(model, "prototypes"))

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[
            vis,
        ],
    )

    # Training loop
    trainer.fit(model, train_loader)
