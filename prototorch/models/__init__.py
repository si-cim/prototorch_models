"""`models` plugin for the `prototorch` package."""

from importlib.metadata import PackageNotFoundError, version

from .callbacks import PrototypeConvergence, PruneLoserPrototypes
from .cbc import CBC, ImageCBC
from .glvq import (
    GLVQ,
    GLVQ1,
    GLVQ21,
    GMLVQ,
    GRLVQ,
    LGMLVQ,
    LVQMLN,
    ImageGLVQ,
    ImageGMLVQ,
    SiameseGLVQ,
    SiameseGMLVQ,
)
from .knn import KNN
from .lvq import LVQ1, LVQ21, MedianLVQ
from .probabilistic import CELVQ, RSLVQ, LikelihoodRatioLVQ
from .unsupervised import GrowingNeuralGas, NeuralGas
from .vis import *

from .oneclass import *

__version__ = "0.1.7"
