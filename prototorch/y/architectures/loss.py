from dataclasses import dataclass, field

from prototorch.core.losses import GLVQLoss
from prototorch.y.architectures.base import BaseYArchitecture


class GLVQLossMixin(BaseYArchitecture):
    """
    GLVQ Loss

    A loss layer that uses the Generalized Learning Vector Quantization (GLVQ) loss.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        margin: The margin of the GLVQ loss. Default: 0.0.
        transfer_fn: Transfer function to use. Default: sigmoid_beta.
        transfer_args: Keyword arguments for the transfer function. Default: {beta: 10.0}.
        """
        margin: float = 0.0

        transfer_fn: str = "sigmoid_beta"
        transfer_args: dict = field(default_factory=lambda: dict(beta=10.0))

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def init_loss(self, hparams: HyperParameters):
        self.loss_layer = GLVQLoss(
            margin=hparams.margin,
            transfer_fn=hparams.transfer_fn,
            **hparams.transfer_args,
        )

    def loss(self, comparison_measures, batch, components):
        target = batch[1]
        comp_labels = components[1]
        loss = self.loss_layer(comparison_measures, target, comp_labels)
        self.log('loss', loss)
        return loss
