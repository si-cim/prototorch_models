"""`models` plugin for the `prototorch` package."""

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
from .probabilistic import CELVQ, PLVQ, RSLVQ, SLVQ
from .unsupervised import GrowingNeuralGas, HeskesSOM, KohonenSOM, NeuralGas
from .vis import *

__version__ = "0.4.0"
