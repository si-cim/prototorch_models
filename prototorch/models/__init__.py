from importlib.metadata import PackageNotFoundError, version

from .cbc import CBC
from .glvq import (GLVQ, GMLVQ, GRLVQ, LVQ1, LVQ21, LVQMLN, ImageGLVQ,
                   SiameseGLVQ)
from .knn import KNN
from .neural_gas import NeuralGas
from .vis import *

__version__ = "0.1.7"
