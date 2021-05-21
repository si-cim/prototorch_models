"""Unsupervised prototype learning algorithms."""

import warnings

import torch
import torchmetrics
from prototorch.components import Components, LabeledComponents
from prototorch.components import initializers as cinit
from prototorch.components.initializers import ZerosInitializer, parse_data_arg
from prototorch.functions.competitions import knnc
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import NeuralGasEnergy

from .abstract import AbstractPrototypeModel


class EuclideanDistance(torch.nn.Module):
    def forward(self, x, y):
        return euclidean_distance(x, y)


class ConnectionTopology(torch.nn.Module):
    def __init__(self, agelimit, num_prototypes):
        super().__init__()
        self.agelimit = agelimit
        self.num_prototypes = num_prototypes

        self.cmat = torch.zeros((self.num_prototypes, self.num_prototypes))
        self.age = torch.zeros_like(self.cmat)

    def forward(self, d):
        order = torch.argsort(d, dim=1)

        for element in order:
            i0, i1 = element[0], element[1]
            self.cmat[i0][i1] = 1
            self.age[i0][i1] = 0
            self.age[i0][self.cmat[i0] == 1] += 1
            self.cmat[i0][self.age[i0] > self.agelimit] = 0

    def extra_repr(self):
        return f"agelimit: {self.agelimit}"


class KNN(AbstractPrototypeModel):
    """K-Nearest-Neighbors classification algorithm."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        # Default Values
        self.hparams.setdefault("k", 1)
        self.hparams.setdefault("distance", euclidean_distance)

        data = kwargs.get("data")
        x_train, y_train = parse_data_arg(data)

        self.proto_layer = LabeledComponents(initialized_components=(x_train,
                                                                     y_train))

        self.train_acc = torchmetrics.Accuracy()

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach()

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
        return y_pred

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


class NeuralGas(AbstractPrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.optimizer = kwargs.get("optimizer", torch.optim.Adam)

        # Default Values
        self.hparams.setdefault("input_dim", 2)
        self.hparams.setdefault("agelimit", 10)
        self.hparams.setdefault("lm", 1)
        self.hparams.setdefault("prototype_initializer",
                                ZerosInitializer(self.hparams.input_dim))

        self.proto_layer = Components(
            self.hparams.num_prototypes,
            initializer=self.hparams.prototype_initializer)

        self.distance_layer = EuclideanDistance()
        self.energy_layer = NeuralGasEnergy(lm=self.hparams.lm)
        self.topology_layer = ConnectionTopology(
            agelimit=self.hparams.agelimit,
            num_prototypes=self.hparams.num_prototypes,
        )

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        protos = self.proto_layer()
        d = self.distance_layer(x, protos)
        cost, order = self.energy_layer(d)

        self.topology_layer(d)
        return cost
