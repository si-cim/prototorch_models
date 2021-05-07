import torch
from prototorch.components import Components
from prototorch.components import initializers as cinit
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


class NeuralGas(AbstractPrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        # Default Values
        self.hparams.setdefault("input_dim", 2)
        self.hparams.setdefault("agelimit", 10)
        self.hparams.setdefault("lm", 1)
        self.hparams.setdefault("prototype_initializer",
                                cinit.ZerosInitializer(self.hparams.input_dim))

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
