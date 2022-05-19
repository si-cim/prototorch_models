from dataclasses import dataclass
from typing import Type

import torch
from prototorch.models.y_arch import BaseYArchitecture


class SingleLearningRateMixin(BaseYArchitecture):
    """
    Single Learning Rate

    All parameters are updated with a single learning rate.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        lr: The learning rate. Default: 0.1.
        optimizer: The optimizer to use. Default: torch.optim.Adam.
        """
        lr: float = 0.1
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def __init__(self, hparams: HyperParameters) -> None:
        super().__init__(hparams)
        self.lr = hparams.lr
        self.optimizer = hparams.optimizer

    # Hooks
    # ----------------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)  # type: ignore
