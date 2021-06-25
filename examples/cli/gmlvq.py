"""GMLVQ example using the MNIST dataset."""

import prototorch as pt
import torch
from prototorch.models import ImageGMLVQ
from prototorch.models.abstract import PrototypeModel
from prototorch.models.data import MNISTDataModule
from pytorch_lightning.utilities.cli import LightningCLI


class ExperimentClass(ImageGMLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         optimizer=torch.optim.Adam,
                         prototype_initializer=pt.components.zeros(28 * 28),
                         **kwargs)


cli = LightningCLI(ImageGMLVQ, MNISTDataModule)
