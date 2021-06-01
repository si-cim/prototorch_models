"""Unsupervised prototype learning algorithms."""

import logging
import warnings

import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.components import Components, LabeledComponents
from prototorch.components.initializers import ZerosInitializer, parse_data_arg
from prototorch.functions.competitions import knnc
from prototorch.functions.distances import euclidean_distance
from prototorch.modules import LambdaLayer
from prototorch.modules.losses import NeuralGasEnergy
from pytorch_lightning.callbacks import Callback

from .abstract import AbstractPrototypeModel


class GNGCallback(Callback):
    """GNG Callback.

    Applies growing algorithm based on accumulated error and topology.

    Based on "A Growing Neural Gas Network Learns Topologies" by Bernd Fritzke.
    """
    def __init__(self, reduction=0.1, freq=10):
        self.reduction = reduction
        self.freq = freq

    def on_epoch_end(self, trainer: pl.Trainer, pl_module):
        if (trainer.current_epoch + 1) % self.freq == 0:
            # Get information
            errors = pl_module.errors
            topology: ConnectionTopology = pl_module.topology_layer
            components: pt.components.Components = pl_module.proto_layer.components

            # Insertion point
            worst = torch.argmax(errors)

            neighbors = topology.get_neighbors(worst)[0]

            if len(neighbors) == 0:
                logging.log(level=20, msg="No neighbor-pairs found!")
                return

            neighbors_errors = errors[neighbors]
            worst_neighbor = neighbors[torch.argmax(neighbors_errors)]

            # New Prototype
            new_component = 0.5 * (components[worst] +
                                   components[worst_neighbor])

            # Add component
            pl_module.proto_layer.add_components(
                initialized_components=new_component)

            # Adjust Topology
            topology.add_prototype()
            topology.add_connection(worst, -1)
            topology.add_connection(worst_neighbor, -1)
            topology.remove_connection(worst, worst_neighbor)

            # New errors
            worst_error = errors[worst].unsqueeze(0)
            pl_module.errors = torch.cat([pl_module.errors, worst_error])
            pl_module.errors[worst] = errors[worst] * self.reduction
            pl_module.errors[
                worst_neighbor] = errors[worst_neighbor] * self.reduction

            trainer.accelerator_backend.setup_optimizers(trainer)


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
            self.cmat[i1][i0] = 1

            self.age[i0][i1] = 0
            self.age[i1][i0] = 0

            self.age[i0][self.cmat[i0] == 1] += 1
            self.age[i1][self.cmat[i1] == 1] += 1

            self.cmat[i0][self.age[i0] > self.agelimit] = 0
            self.cmat[i1][self.age[i1] > self.agelimit] = 0

    def get_neighbors(self, position):
        return torch.where(self.cmat[position])

    def add_prototype(self):
        new_cmat = torch.zeros([dim + 1 for dim in self.cmat.shape])
        new_cmat[:-1, :-1] = self.cmat
        self.cmat = new_cmat

        new_age = torch.zeros([dim + 1 for dim in self.age.shape])
        new_age[:-1, :-1] = self.age
        self.age = new_age

    def add_connection(self, a, b):
        self.cmat[a][b] = 1
        self.cmat[b][a] = 1

        self.age[a][b] = 0
        self.age[b][a] = 0

    def remove_connection(self, a, b):
        self.cmat[a][b] = 0
        self.cmat[b][a] = 0

        self.age[a][b] = 0
        self.age[b][a] = 0

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

        self.distance_layer = LambdaLayer(euclidean_distance)
        self.energy_layer = NeuralGasEnergy(lm=self.hparams.lm)
        self.topology_layer = ConnectionTopology(
            agelimit=self.hparams.agelimit,
            num_prototypes=self.hparams.num_prototypes,
        )

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        protos = self.proto_layer()
        d = self.distance_layer(x, protos)
        cost, _ = self.energy_layer(d)
        self.topology_layer(d)
        return cost


class GrowingNeuralGas(NeuralGas):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # defaults
        self.hparams.setdefault("step_reduction", 0.5)
        self.hparams.setdefault("insert_reduction", 0.1)
        self.hparams.setdefault("insert_freq", 10)

        self.register_buffer("errors",
                             torch.zeros(self.hparams.num_prototypes))

    def training_step(self, train_batch, _batch_idx):
        x = train_batch[0]
        protos = self.proto_layer()
        d = self.distance_layer(x, protos)
        cost, order = self.energy_layer(d)
        winner = order[:, 0]
        mask = torch.zeros_like(d)
        mask[torch.arange(len(mask)), winner] = 1.0
        winner_distances = d * mask

        self.errors += torch.sum(winner_distances * winner_distances, dim=0)
        self.errors *= self.hparams.step_reduction

        self.topology_layer(d)
        return cost

    def configure_callbacks(self):
        return [
            GNGCallback(reduction=self.hparams.insert_reduction,
                        freq=self.hparams.insert_freq)
        ]
