from importlib.metadata import PackageNotFoundError, version

from .cbc import CBC
from .glvq import (GLVQ, GMLVQ, GRLVQ, GLVQ1, GLVQ21, LVQ1, LVQ21, LVQMLN,
                   ImageGLVQ, ImageGMLVQ, SiameseGLVQ)
from .knn import KNN
from .neural_gas import NeuralGas
from .vis import *

__version__ = "0.1.7"
