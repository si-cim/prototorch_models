from typing import Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.models.proto_y_architecture.base import BaseYArchitecture
from prototorch.models.vis import Vis2DAbstract
from prototorch.utils.utils import mesh2d


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
                self.name,
                self.metric,
                **self.metric_kwargs,
            )
        else:
            raise ValueError(f"{self.on} is no valid metric hook")


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
