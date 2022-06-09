import warnings
from typing import Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from matplotlib import pyplot as plt
from prototorch.models.vis import Vis2DAbstract
from prototorch.utils.utils import mesh2d
from prototorch.y.architectures.base import BaseYArchitecture
from prototorch.y.library.gmlvq import GMLVQ
from pytorch_lightning.loggers import TensorBoardLogger

DIVERGING_COLOR_MAPS = [
    'PiYG',
    'PRGn',
    'BrBG',
    'PuOr',
    'RdGy',
    'RdBu',
    'RdYlBu',
    'RdYlGn',
    'Spectral',
    'coolwarm',
    'bwr',
    'seismic',
]


class LogTorchmetricCallback(pl.Callback):

    def __init__(
        self,
        name,
        metric: Type[torchmetrics.Metric],
        on="prediction",
        **metric_kwargs,
    ) -> None:
        self.name = name
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.on = on

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: BaseYArchitecture,
        stage: Optional[str] = None,
    ) -> None:
        if self.on == "prediction":
            pl_module.register_torchmetric(
                self,
                self.metric,
                **self.metric_kwargs,
            )
        else:
            raise ValueError(f"{self.on} is no valid metric hook")

    def __call__(self, value, pl_module: BaseYArchitecture):
        pl_module.log(self.name, value)


class LogConfusionMatrix(LogTorchmetricCallback):

    def __init__(
        self,
        num_classes,
        name="confusion",
        on='prediction',
        **kwargs,
    ):
        super().__init__(
            name,
            torchmetrics.ConfusionMatrix,
            on=on,
            num_classes=num_classes,
            **kwargs,
        )

    def __call__(self, value, pl_module: BaseYArchitecture):
        fig, ax = plt.subplots()
        ax.imshow(value.detach().cpu().numpy())

        # Show all ticks and label them with the respective list entries
        # ax.set_xticks(np.arange(len(farmers)), labels=farmers)
        # ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

        # Rotate the tick labels and set their alignment.
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        # Loop over data dimensions and create text annotations.
        for i in range(len(value)):
            for j in range(len(value)):
                text = ax.text(
                    j,
                    i,
                    value[i, j].item(),
                    ha="center",
                    va="center",
                    color="w",
                )

        ax.set_title(self.name)
        fig.tight_layout()

        pl_module.logger.experiment.add_figure(
            tag=self.name,
            figure=fig,
            close=True,
            global_step=pl_module.global_step,
        )


class VisGLVQ2D(Vis2DAbstract):

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        ax = self.setup_ax()
        self.plot_protos(ax, protos, plabels)
        if x_train is not None:
            self.plot_data(ax, x_train, y_train)
            mesh_input, xx, yy = mesh2d(
                np.vstack([x_train, protos]),
                self.border,
                self.resolution,
            )
        else:
            mesh_input, xx, yy = mesh2d(protos, self.border, self.resolution)
        _components = pl_module.components_layer.components
        mesh_input = torch.from_numpy(mesh_input).type_as(_components)
        y_pred = pl_module.predict(mesh_input)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)


class VisGMLVQ2D(Vis2DAbstract):

    def __init__(self, *args, ev_proj=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_proj = ev_proj

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        device = pl_module.device
        omega = pl_module._omega.detach()
        lam = omega @ omega.T
        u, _, _ = torch.pca_lowrank(lam, q=2)
        with torch.no_grad():
            x_train = torch.Tensor(x_train).to(device)
            x_train = x_train @ u
            x_train = x_train.cpu().detach()
        if self.show_protos:
            with torch.no_grad():
                protos = torch.Tensor(protos).to(device)
                protos = protos @ u
                protos = protos.cpu().detach()
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        if self.show_protos:
            self.plot_protos(ax, protos, plabels)


class PlotLambdaMatrixToTensorboard(pl.Callback):

    def __init__(self, cmap='seismic') -> None:
        super().__init__()
        self.cmap = cmap

        if self.cmap not in DIVERGING_COLOR_MAPS and type(self.cmap) is str:
            warnings.warn(
                f"{self.cmap} is not a diverging color map. We recommend to use one of the following: {DIVERGING_COLOR_MAPS}"
            )

    def on_train_start(self, trainer, pl_module: GMLVQ):
        self.plot_lambda(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module: GMLVQ):
        self.plot_lambda(trainer, pl_module)

    def plot_lambda(self, trainer, pl_module: GMLVQ):

        self.fig, self.ax = plt.subplots(1, 1)

        # plot lambda matrix
        l_matrix = pl_module.lambda_matrix

        # normalize lambda matrix
        l_matrix = l_matrix / torch.max(torch.abs(l_matrix))

        # plot lambda matrix
        self.ax.imshow(l_matrix.detach().numpy(), self.cmap, vmin=-1, vmax=1)

        self.fig.colorbar(self.ax.images[-1])

        # add title
        self.ax.set_title('Lambda Matrix')

        # add to tensorboard
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_figure(
                f"lambda_matrix",
                self.fig,
                trainer.global_step,
            )
        else:
            warnings.warn(
                f"{self.__class__.__name__} is not compatible with {trainer.logger.__class__.__name__} as logger. Use TensorBoardLogger instead."
            )
