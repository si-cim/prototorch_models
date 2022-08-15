from dataclasses import dataclass

from prototorch.core.competitions import WTAC
from prototorch.models.architectures.base import BaseYArchitecture


class WTACompetitionMixin(BaseYArchitecture):
    """
    Winner Take All Competition

    A competition layer that uses the winner-take-all strategy.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        No hyperparameters.
        """

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def init_inference(self, hparams: HyperParameters):
        self.competition_layer = WTAC()

    def inference(self, comparison_measures, components):
        comp_labels = components[1]
        return self.competition_layer(comparison_measures, comp_labels)
