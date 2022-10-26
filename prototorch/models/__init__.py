from .architectures.base import BaseYArchitecture
from .architectures.comparison import (
    OmegaComparisonMixin,
    SimpleComparisonMixin,
)
from .architectures.competition import WTACompetitionMixin
from .architectures.components import SupervisedArchitecture
from .architectures.loss import GLVQLossMixin
from .architectures.optimization import (
    MultipleLearningRateMixin,
    SingleLearningRateMixin,
)

__all__ = [
    'BaseYArchitecture',
    "OmegaComparisonMixin",
    "SimpleComparisonMixin",
    "SingleLearningRateMixin",
    "MultipleLearningRateMixin",
    "SupervisedArchitecture",
    "WTACompetitionMixin",
    "GLVQLossMixin",
]

__version__ = "1.0.0-a8"
