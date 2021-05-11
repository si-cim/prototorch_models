"""The popular K-Nearest-Neighbors classification algorithm."""

import warnings

import torch
import torchmetrics
from prototorch.components import LabeledComponents
from prototorch.components.initializers import parse_init_arg
from prototorch.functions.competitions import knnc
from prototorch.functions.distances import euclidean_distance

from .abstract import AbstractPrototypeModel


class KNN(AbstractPrototypeModel):
    """K-Nearest-Neighbors classification algorithm."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        # Default Values
        self.hparams.setdefault("k", 1)
        self.hparams.setdefault("distance", euclidean_distance)

        data = kwargs.get("data")
        x_train, y_train = parse_init_arg(data)

        self.proto_layer = LabeledComponents(initialized_components=(x_train,
                                                                     y_train))

        self.train_acc = torchmetrics.Accuracy()

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach().cpu()

    def forward(self, x):
        protos, _ = self.proto_layer()
        dis = self.hparams.distance(x, protos)
        return dis

    def predict(self, x):
        # model.eval()  # ?!
        with torch.no_grad():
            d = self(x)
            plabels = self.proto_layer.component_labels
            y_pred = knnc(d, plabels, k=self.hparams.k)
        return y_pred.numpy()

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        return 1

    def on_train_batch_start(self,
                             train_batch,
                             batch_idx,
                             dataloader_idx=None):
        warnings.warn("k-NN has no training, skipping!")
        return -1

    def configure_optimizers(self):
        return None
