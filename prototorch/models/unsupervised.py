"""Unsupervised prototype learning algorithms."""

import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.components import Components, LabeledComponents
from prototorch.components.initializers import ZerosInitializer
from prototorch.functions.competitions import knnc
from prototorch.functions.distances import euclidean_distance
from prototorch.modules import LambdaLayer
from prototorch.modules.losses import NeuralGasEnergy
from pytorch_lightning.callbacks import Callback

from .abstract import UnsupervisedPrototypeModel
from .callbacks import GNGCallback
from .extras import ConnectionTopology


class NeuralGas(UnsupervisedPrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Hyperparameters
        self.save_hyperparameters(hparams)

        # Default hparams
        self.hparams.setdefault("agelimit", 10)
        self.hparams.setdefault("lm", 1)

        self.energy_layer = NeuralGasEnergy(lm=self.hparams.lm)
        self.topology_layer = ConnectionTopology(
            agelimit=self.hparams.agelimit,
            num_prototypes=self.hparams.num_prototypes,
        )

    def training_step(self, train_batch, batch_idx):
        # x = train_batch
        # TODO Check if the batch has labels
        x = train_batch[0]
        d = self.compute_distances(x)
        cost, _ = self.energy_layer(d)
        self.topology_layer(d)
        return cost


class GrowingNeuralGas(NeuralGas):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Defaults
        self.hparams.setdefault("step_reduction", 0.5)
        self.hparams.setdefault("insert_reduction", 0.1)
        self.hparams.setdefault("insert_freq", 10)

        errors = torch.zeros(self.hparams.num_prototypes, device=self.device)
        self.register_buffer("errors", errors)

    def training_step(self, train_batch, _batch_idx):
        # x = train_batch
        # TODO Check if the batch has labels
        x = train_batch[0]
        d = self.compute_distances(x)
        cost, order = self.energy_layer(d)
        winner = order[:, 0]
        mask = torch.zeros_like(d)
        mask[torch.arange(len(mask)), winner] = 1.0
        dp = d * mask

        self.errors += torch.sum(dp * dp, dim=0)
        self.errors *= self.hparams.step_reduction

        self.topology_layer(d)
        return cost

    def configure_callbacks(self):
        return [
            GNGCallback(reduction=self.hparams.insert_reduction,
                        freq=self.hparams.insert_freq)
        ]
