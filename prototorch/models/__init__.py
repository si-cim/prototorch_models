from importlib.metadata import PackageNotFoundError, version

from . import probabilistic
from .cbc import CBC, ImageCBC
from .glvq import (GLVQ, GLVQ1, GLVQ21, GMLVQ, GRLVQ, LVQ1, LVQ21, LVQMLN,
                   ImageGLVQ, ImageGMLVQ, SiameseGLVQ)
from .unsupervised import KNN, NeuralGas
from .vis import *

__version__ = "0.1.7"
