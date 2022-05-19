from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from prototorch.core.distances import omega_distance
from prototorch.core.initializers import (
    AbstractLinearTransformInitializer,
    EyeLinearTransformInitializer,
)
from prototorch.models.y_arch import (
    GLVQLossMixin,
    SimpleComparisonMixin,
    SupervisedArchitecture,
    WTACompetitionMixin,
)
from prototorch.nn.wrappers import LambdaLayer
from torch.nn.parameter import Parameter


class GMLVQ(
        SupervisedArchitecture,
        SimpleComparisonMixin,
        GLVQLossMixin,
        WTACompetitionMixin,
):
    """
    Generalized Matrix Learning Vector Quantization (GMLVQ)

    A GMLVQ architecture that uses the winner-take-all strategy and the GLVQ loss.
    """

    _omega: torch.Tensor

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(
            SimpleComparisonMixin.HyperParameters,
            GLVQLossMixin.HyperParameters,
            WTACompetitionMixin.HyperParameters,
            SupervisedArchitecture.HyperParameters,
    ):
        """
        comparison_fn: The comparison / dissimilarity function to use. Override Default: omega_distance.
        comparison_args: Keyword arguments for the comparison function. Override Default: {}.
        input_dim: Necessary Field: The dimensionality of the input.
        latent_dim: The dimensionality of the latent space. Default: 2.
        omega_initializer: The initializer to use for the omega matrix. Default: EyeLinearTransformInitializer.
        """
        backbone_lr: float = 0.1
        lr: float = 0.1
        comparison_fn: Callable = omega_distance
        comparison_args: dict = field(default_factory=lambda: dict())
        input_dim: int | None = None
        latent_dim: int = 2
        omega_initializer: type[
            AbstractLinearTransformInitializer] = EyeLinearTransformInitializer

        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, hparams) -> None:
        super().__init__(hparams)
        self.lr = hparams.lr
        self.backbone_lr = hparams.backbone_lr
        self.optimizer = hparams.optimizer

    def init_comparison(self, hparams: HyperParameters) -> None:
        if hparams.input_dim is None:
            raise ValueError("input_dim must be specified.")
        omega = hparams.omega_initializer().generate(
            hparams.input_dim,
            hparams.latent_dim,
        )
        self.register_parameter("_omega", Parameter(omega))
        self.comparison_layer = LambdaLayer(
            fn=hparams.comparison_fn,
            **hparams.comparison_args,
        )

    def comparison(self, batch, components):
        comp_tensor, _ = components
        batch_tensor, _ = batch

        comp_tensor = comp_tensor.unsqueeze(1)

        distances = self.comparison_layer(
            batch_tensor,
            comp_tensor,
            self._omega,
        )

        return distances

    def configure_optimizers(self):
        proto_opt = self.optimizer(
            self.components_layer.parameters(),
            lr=self.lr,
        )
        omega_opt = self.optimizer(
            [self._omega],
            lr=self.backbone_lr,
        )
        return [proto_opt, omega_opt]

    # Properties
    # ----------------------------------------------------------------------------------------------------
    @property
    def omega_matrix(self):
        return self._omega.detach().cpu()

    @property
    def lambda_matrix(self):
        omega = self._omega.detach()
        lam = omega @ omega.T
        return lam.detach().cpu()
