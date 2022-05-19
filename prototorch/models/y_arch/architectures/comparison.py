from dataclasses import dataclass, field
from typing import Callable

from prototorch.core.distances import euclidean_distance
from prototorch.models.y_arch.architectures.base import BaseYArchitecture
from prototorch.nn.wrappers import LambdaLayer


class SimpleComparisonMixin(BaseYArchitecture):
    """
    Simple Comparison

    A comparison layer that only uses the positions of the components and the batch for dissimilarity computation.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        comparison_fn: The comparison / dissimilarity function to use. Default: euclidean_distance.
        comparison_args: Keyword arguments for the comparison function. Default: {}.
        """
        comparison_fn: Callable = euclidean_distance
        comparison_args: dict = field(default_factory=lambda: dict())

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def init_comparison(self, hparams: HyperParameters):
        self.comparison_layer = LambdaLayer(fn=hparams.comparison_fn,
                                            **hparams.comparison_args)

    def comparison(self, batch, components):
        comp_tensor, _ = components
        batch_tensor, _ = batch

        comp_tensor = comp_tensor.unsqueeze(1)

        distances = self.comparison_layer(batch_tensor, comp_tensor)

        return distances
