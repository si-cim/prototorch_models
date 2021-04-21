"""GLVQ example using the Iris dataset."""

import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from prototorch.models.glvq import GLVQ
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset


class NumpyDataset(TensorDataset):
    def __init__(self, *arrays):
        tensors = [torch.from_numpy(arr) for arr in arrays]
        super().__init__(*tensors)


class GLVQIris(GLVQ):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-1)
        parser.add_argument("--batch_size", type=int, default=150)
        parser.add_argument("--input_dim", type=int, default=2)
        parser.add_argument("--nclasses", type=int, default=3)
        parser.add_argument("--prototypes_per_class", type=int, default=3)
        parser.add_argument("--prototype_initializer",
                            type=str,
                            default="stratified_mean")
        return parser


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
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel("Data dimension 1")
        ax.set_ylabel("Data dimension 2")
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
        ax.scatter(protos[:, 0],
                   protos[:, 1],
                   c=plabels,
                   cmap=self.cmap,
                   edgecolor="k",
                   marker="D",
                   s=50)
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
    # For best-practices when using `argparse` with `pytorch_lightning`, see
    # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html
    parser = argparse.ArgumentParser()

    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Add model specific args
    parser = GLVQIris.add_model_specific_args(parser)

    # Callbacks
    vis = VisualizationCallback(x_train, y_train)

    # Automatically add trainer-specific-args like `--gpus`, `--num_nodes` etc.
    parser = pl.Trainer.add_argparse_args(parser)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        parser,
        max_epochs=10,
        callbacks=[
            vis,  # comment this line out to disable the visualization
        ],
    )
    # trainer.tune(model)

    # Initialize the model
    args = parser.parse_args()
    model = GLVQIris(args, data=[x_train, y_train])

    # Model summary
    print(model)

    # Training loop
    trainer.fit(model, train_loader)

    # Save the model manually (use `pl.callbacks.ModelCheckpoint` to automate)
    ckpt = "glvq_iris.ckpt"
    trainer.save_checkpoint(ckpt)

    # Load the checkpoint
    new_model = GLVQIris.load_from_checkpoint(checkpoint_path=ckpt)

    print(new_model)
