from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from prototorch.core.distances import euclidean_distance
from prototorch.core.initializers import (
    AbstractLinearTransformInitializer,
    EyeLinearTransformInitializer,
)
from prototorch.models.architectures.base import BaseYArchitecture
from prototorch.nn.wrappers import LambdaLayer
from torch import Tensor
from torch.nn.parameter import Parameter


class SimpleComparisonMixin(BaseYArchitecture):
    """
    Simple Comparison

    A comparison layer that only uses the positions of the components
    and the batch for dissimilarity computation.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        comparison_fn: The comparison / dissimilarity function to use. Default: euclidean_distance.
        comparison_args: Keyword arguments for the comparison function. Default: {}.
        """
        comparison_fn: Callable = euclidean_distance
        comparison_args: dict = field(default_factory=dict)

        comparison_parameters: dict = field(default_factory=dict)

    # Steps
    # ----------------------------------------------------------------------------------------------
    def init_comparison(self, hparams: HyperParameters):
        self.comparison_layer = LambdaLayer(
            fn=hparams.comparison_fn,
            **hparams.comparison_args,
        )

        self.comparison_kwargs: dict[str, Tensor] = {}

    def comparison(self, batch, components):
        comp_tensor, _ = components
        batch_tensor, _ = batch

        comp_tensor = comp_tensor.unsqueeze(1)

        distances = self.comparison_layer(
            batch_tensor,
            comp_tensor,
            **self.comparison_kwargs,
        )

        return distances


class OmegaComparisonMixin(SimpleComparisonMixin):
    """
    Omega Comparison

    A comparison layer that uses the positions of the components
    and the batch for dissimilarity computation.
    """

    _omega: torch.Tensor

    # HyperParameters
    # ----------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(SimpleComparisonMixin.HyperParameters):
        """
        input_dim: Necessary Field: The dimensionality of the input.
        latent_dim:
            The dimensionality of the latent space. Default: 2.
        omega_initializer:
            The initializer to use for the omega matrix. Default: EyeLinearTransformInitializer.
        """
        input_dim: int | None = None
        latent_dim: int = 2
        omega_initializer: type[
            AbstractLinearTransformInitializer] = EyeLinearTransformInitializer
        omega_initializer_kwargs: dict = field(default_factory=dict)

    # Steps
    # ----------------------------------------------------------------------------------------------
    def init_comparison(self, hparams: HyperParameters) -> None:
        super().init_comparison(hparams)

        # Initialize the omega matrix
        if hparams.input_dim is None:
            raise ValueError("input_dim must be specified.")
        else:
            omega = hparams.omega_initializer(
                **hparams.omega_initializer_kwargs).generate(
                    hparams.input_dim,
                    hparams.latent_dim,
                )
            self.register_parameter("_omega", Parameter(omega))
            self.comparison_kwargs = dict(omega=self._omega)

    # Properties
    # ----------------------------------------------------------------------------------------------
    @property
    def omega_matrix(self):
        '''
        Omega Matrix. Mapping applied to data and prototypes.
        '''
        return self._omega.detach().cpu()

    @property
    def lambda_matrix(self):
        '''
        Lambda Matrix.
        '''
        omega = self._omega.detach()
        lam = omega @ omega.T
        return lam.detach().cpu()

    @property
    def relevance_profile(self):
        '''
        Relevance Profile. Main Diagonal of the Lambda Matrix.
        '''
        return self.lambda_matrix.diag().abs()

    @property
    def classification_influence_profile(self):
        '''
        Classification Influence Profile. Influence of each dimension.
        '''
        lam = self.lambda_matrix
        return lam.abs().sum(0)

    @property
    def parameter_omega(self):
        return self._omega

    @parameter_omega.setter
    def parameter_omega(self, new_omega):
        with torch.no_grad():
            self._omega.data.copy_(new_omega)
