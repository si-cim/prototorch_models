from importlib.metadata import PackageNotFoundError, version

from .cbc import CBC
from .glvq import GLVQ, GMLVQ, GRLVQ, LVQMLN, ImageGLVQ, SiameseGLVQ
from .neural_gas import NeuralGas
from .vis import *

VERSION_FALLBACK = "uninstalled_version"
try:
    __version__ = version(__name__.replace(".", "-"))
except PackageNotFoundError:
    __version__ = VERSION_FALLBACK
    pass
