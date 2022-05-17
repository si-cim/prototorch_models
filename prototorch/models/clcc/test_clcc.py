from typing import Optional, Type

import numpy as np
import prototorch as pt
import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.core import SMCI
from prototorch.models.clcc.clcc_glvq import GLVQ
from prototorch.models.clcc.clcc_scheme import CLCCScheme
from prototorch.models.vis import Vis2DAbstract
from prototorch.utils.utils import mesh2d
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

# NEW STUFF
# ##############################################################################


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
        pl_module: CLCCScheme,
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


# TODO: Pruning

# ##############################################################################

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_ds.targets[train_ds.targets == 2.0] = 1.0
    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        num_workers=0,
        shuffle=True,
    )

    components_initializer = SMCI(train_ds)
    #components_initializer = RandomNormalCompInitializer(2)

    hyperparameters = GLVQ.HyperParameters(
        lr=0.5,
        distribution=dict(
            num_classes=2,
            per_class=1,
        ),
        component_initializer=components_initializer,
    )

    model = GLVQ(hyperparameters)

    print(model)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)
    recall = LogTorchmetricCallback(
        'recall',
        torchmetrics.Recall,
        num_classes=2,
    )

    es = EarlyStopping(
        monitor="recall",
        min_delta=0.001,
        patience=15,
        mode="max",
        check_on_train_epoch_end=True,
    )

    # Train
    trainer = pl.Trainer(
        callbacks=[
            vis,
            recall,
            es,
        ],
        gpus=0,
        max_epochs=200,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader)
