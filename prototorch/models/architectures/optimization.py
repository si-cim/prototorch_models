from dataclasses import dataclass, field
from typing import Type

import torch
from prototorch.models import BaseYArchitecture
from torch.nn.parameter import Parameter


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

    # Hooks
    # ----------------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        return self.hparams.optimizer(self.parameters(),
                                      lr=self.hparams.lr)  # type: ignore


class MultipleLearningRateMixin(BaseYArchitecture):
    """
    Multiple Learning Rates

    Define Different Learning Rates for different parameters.
    """

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        """
        lr: The learning rate. Default: 0.1.
        optimizer: The optimizer to use. Default: torch.optim.Adam.
        """
        lr: dict = field(default_factory=dict)
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam

    # Hooks
    # ----------------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizers = []
        for name, lr in self.hparams.lr.items():
            if not hasattr(self, name):
                raise ValueError(f"{name} is not a parameter of {self}")
            else:
                model_part = getattr(self, name)
                if isinstance(model_part, Parameter):
                    optimizers.append(
                        self.hparams.optimizer(
                            [model_part],
                            lr=lr,  # type: ignore
                        ))
                elif hasattr(model_part, "parameters"):
                    optimizers.append(
                        self.hparams.optimizer(
                            model_part.parameters(),
                            lr=lr,  # type: ignore
                        ))
        return optimizers
