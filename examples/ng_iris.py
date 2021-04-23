"""CBC example using the Iris dataset."""

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.neural_gas import NeuralGas


class VisualizationCallback(pl.Callback):
    def __init__(self,
                 x_train,
                 y_train,
                 title="Neural Gas Visualization",
                 cmap="viridis"):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.title = title
        self.fig = plt.figure(self.title)
        self.cmap = cmap

    def on_epoch_end(self, trainer, pl_module: NeuralGas):
        protos = pl_module.proto_layer.prototypes.detach().cpu().numpy()
        cmat = pl_module.topology_layer.cmat.cpu().numpy()

        # Visualize the data and the prototypes
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel("Data dimension 1")
        ax.set_ylabel("Data dimension 2")
        ax.scatter(self.x_train[:, 0],
                   self.x_train[:, 1],
                   c=self.y_train,
                   edgecolor="k")
        ax.scatter(
            protos[:, 0],
            protos[:, 1],
            c="k",
            edgecolor="k",
            marker="D",
            s=50,
        )

        # Draw connections
        for i in range(len(protos)):
            for j in range(len(protos)):
                if cmat[i][j]:
                    ax.plot(
                        [protos[i, 0], protos[j, 0]],
                        [protos[i, 1], protos[j, 1]],
                        "k-",
                    )

        plt.pause(0.01)


if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    y_single_class = np.zeros_like(y_train)
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        input_dim=x_train.shape[1],
        nclasses=1,
        prototypes_per_class=30,
        prototype_initializer="rand",
        lr=0.01,
    )

    # Initialize the model
    model = NeuralGas(hparams, data=[x_train, y_single_class])

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
