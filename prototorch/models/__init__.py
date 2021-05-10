from importlib.metadata import PackageNotFoundError, version

from .cbc import CBC
from .glvq import GLVQ, GMLVQ, GRLVQ, LVQMLN, ImageGLVQ, SiameseGLVQ
from .neural_gas import NeuralGas
from .vis import *

__version__ = "0.1.4"