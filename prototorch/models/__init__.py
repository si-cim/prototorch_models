"""`models` plugin for the `prototorch` package."""

from .callbacks import PrototypeConvergence, PruneLoserPrototypes
from .cbc import CBC, ImageCBC
from .glvq import (
    GLVQ,
    GLVQ1,
    GLVQ21,
    GMLVQ,
    GRLVQ,
    GTLVQ,
    LGMLVQ,
    LVQMLN,
    ImageGLVQ,
    ImageGMLVQ,
    ImageGTLVQ,
    SiameseGLVQ,
    SiameseGMLVQ,
    SiameseGTLVQ,
)
from .knn import KNN
from .lvq import (
    LVQ1,
    LVQ21,
    MedianLVQ,
)
from .probabilistic import (
    CELVQ,
    RSLVQ,
    SLVQ,
)
from .unsupervised import (
    GrowingNeuralGas,
    KohonenSOM,
    NeuralGas,
)
from .vis import *

__version__ = "0.5.4"
