from typing import Optional

import matplotlib.pyplot as plt
import prototorch as pt
import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.core.initializers import SMCI, RandomNormalCompInitializer
from prototorch.models.clcc.clcc_glvq import GLVQ, GLVQhparams
from prototorch.models.clcc.clcc_scheme import CLCCScheme
from prototorch.models.vis import Visualize2DVoronoiCallback

# NEW STUFF
# ##############################################################################


# TODO: Metrics
class MetricsTestCallback(pl.Callback):
    metric_name = "test_cb_acc"

    def setup(self,
              trainer: pl.Trainer,
              pl_module: CLCCScheme,
              stage: Optional[str] = None) -> None:
        pl_module.register_torchmetric(self.metric_name, torchmetrics.Accuracy)

    def on_epoch_end(self, trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        metric = trainer.logged_metrics[self.metric_name]
        if metric > 0.95:
            trainer.should_stop = True


class LogTorchmetricCallback(pl.Callback):

    def __init__(self, name, metric, on="prediction", **metric_args) -> None:
        self.name = name
        self.metric = metric
        self.metric_args = metric_args
        self.on = on

    def setup(self,
              trainer: pl.Trainer,
              pl_module: CLCCScheme,
              stage: Optional[str] = None) -> None:
        if self.on == "prediction":
            pl_module.register_torchmetric(self.name, self.metric,
                                           **self.metric_args)
        else:
            raise ValueError(f"{self.on} is no valid metric hook")


# TODO: Pruning

# ##############################################################################

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_ds.targets[train_ds.targets == 2.0] = 1.0
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=64,
                                               num_workers=0,
                                               shuffle=True)

    #components_initializer = SMCI(train_ds)
    components_initializer = RandomNormalCompInitializer(2)

    hparams = GLVQhparams(
        lr=0.5,
        distribution=dict(
            num_classes=2,
            per_class=1,
        ),
        component_initializer=components_initializer,
    )
    model = GLVQ(hparams)

    print(model)
    # Callbacks
    vis = Visualize2DVoronoiCallback(
        data=train_ds,
        resolution=500,
    )
    metrics = MetricsTestCallback()
    recall = LogTorchmetricCallback('recall',
                                    torchmetrics.Recall,
                                    num_classes=2)

    # Train
    trainer = pl.Trainer(
        callbacks=[
            vis,
            #metrics,
            recall,
        ],
        gpus=0,
        max_epochs=200,
        weights_summary=None,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader)
