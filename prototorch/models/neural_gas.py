import pytorch_lightning as pl
import torch

from prototorch.functions.distances import euclidean_distance
from prototorch.modules import Prototypes1D
from prototorch.modules.losses import NeuralGasEnergy


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


class NeuralGas(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        # Default Values
        self.hparams.setdefault("agelimit", 10)
        self.hparams.setdefault("lm", 1)
        self.hparams.setdefault("prototype_initializer", "zeros")

        self.proto_layer = Prototypes1D(
            input_dim=self.hparams.input_dim,
            nclasses=self.hparams.nclasses,
            prototypes_per_class=self.hparams.prototypes_per_class,
            prototype_initializer=self.hparams.prototype_initializer,
            **kwargs,
        )

        self.distance_layer = EuclideanDistance()
        self.energy_layer = NeuralGasEnergy(lm=self.hparams.lm)
        self.topology_layer = ConnectionTopology(
            agelimit=self.hparams.agelimit,
            num_prototypes=len(self.proto_layer.prototypes),
        )

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        protos, _ = self.proto_layer()
        d = self.distance_layer(x, protos)
        cost, order = self.energy_layer(d)

        self.topology_layer(d)
        return cost

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
