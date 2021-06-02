"""GMLVQ example using the MNIST dataset."""

from prototorch.models import ImageGMLVQ
from prototorch.models.data import train_on_mnist
from pytorch_lightning.utilities.cli import LightningCLI


class GMLVQMNIST(train_on_mnist(batch_size=64), ImageGMLVQ):
    """Model Definition."""


cli = LightningCLI(GMLVQMNIST)
