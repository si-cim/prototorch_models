"""GMLVQ example using the MNIST dataset."""

from prototorch.models import ImageGLVQ
from pytorch_lightning.utilities.cli import LightningCLI

from mnist import TrainOnMNIST


class Model(TrainOnMNIST, ImageGLVQ):
    """Model Definition"""


cli = LightningCLI(Model)
