from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from prototorch.core.distances import omega_distance
from prototorch.y import (
    GLVQLossMixin,
    MultipleLearningRateMixin,
    OmegaComparisonMixin,
    SupervisedArchitecture,
    WTACompetitionMixin,
)


class GMLVQ(
        SupervisedArchitecture,
        OmegaComparisonMixin,
        GLVQLossMixin,
        WTACompetitionMixin,
        MultipleLearningRateMixin,
):
    """
    Generalized Matrix Learning Vector Quantization (GMLVQ)

    A GMLVQ architecture that uses the winner-take-all strategy and the GLVQ loss.
    """
    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(
            MultipleLearningRateMixin.HyperParameters,
            OmegaComparisonMixin.HyperParameters,
            GLVQLossMixin.HyperParameters,
            WTACompetitionMixin.HyperParameters,
            SupervisedArchitecture.HyperParameters,
    ):
        """
        comparison_fn: The comparison / dissimilarity function to use. Override Default: omega_distance.
        comparison_args: Keyword arguments for the comparison function. Override Default: {}.
        """
        comparison_fn: Callable = omega_distance
        comparison_args: dict = field(default_factory=lambda: dict())
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam

        lr: dict = field(default_factory=lambda: dict(
            components_layer=0.1,
            _omega=0.5,
        ))
