from dataclasses import dataclass

from prototorch.models.y_arch import (
    SimpleComparisonMixin,
    SingleLearningRateMixin,
    SupervisedArchitecture,
    WTACompetitionMixin,
)
from prototorch.models.y_arch.architectures.loss import GLVQLossMixin


class GLVQ(
        SupervisedArchitecture,
        SimpleComparisonMixin,
        GLVQLossMixin,
        WTACompetitionMixin,
        SingleLearningRateMixin,
):
    """
    Generalized Learning Vector Quantization (GLVQ)

    A GLVQ architecture that uses the winner-take-all strategy and the GLVQ loss.
    """

    @dataclass
    class HyperParameters(
            SimpleComparisonMixin.HyperParameters,
            SingleLearningRateMixin.HyperParameters,
            GLVQLossMixin.HyperParameters,
            WTACompetitionMixin.HyperParameters,
            SupervisedArchitecture.HyperParameters,
    ):
        """
        No hyperparameters.
        """
