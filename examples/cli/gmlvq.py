"""GMLVQ example using the MNIST dataset."""

import torch
from pytorch_lightning.utilities.cli import LightningCLI

import prototorch as pt
from prototorch.models import ImageGMLVQ
from prototorch.models.abstract import PrototypeModel
from prototorch.models.data import MNISTDataModule


class ExperimentClass(ImageGMLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         optimizer=torch.optim.Adam,
                         prototype_initializer=pt.components.zeros(28 * 28),
                         **kwargs)


cli = LightningCLI(ImageGMLVQ, MNISTDataModule)
