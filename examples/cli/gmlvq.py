"""GLVQ example using the MNIST dataset."""

from prototorch.models import ImageGLVQ
from prototorch.models.data import train_on_mnist
from pytorch_lightning.utilities.cli import LightningCLI


class GLVQMNIST(train_on_mnist(batch_size=64), ImageGLVQ):
    """Model Definition."""


cli = LightningCLI(GLVQMNIST)
