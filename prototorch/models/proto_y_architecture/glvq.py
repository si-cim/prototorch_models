from dataclasses import dataclass, field
from typing import Callable, Type

import torch
from prototorch.core.competitions import WTAC
from prototorch.core.components import LabeledComponents
from prototorch.core.distances import euclidean_distance
from prototorch.core.initializers import (
    AbstractComponentsInitializer,
    LabelsInitializer,
)
from prototorch.core.losses import GLVQLoss
from prototorch.models.proto_y_architecture.base import BaseYArchitecture
from prototorch.nn.wrappers import LambdaLayer


class SupervisedArchitecture(BaseYArchitecture):
    components_layer: LabeledComponents

    @dataclass
    class HyperParameters:
        distribution: dict[str, int]
        component_initializer: AbstractComponentsInitializer

    def init_components(self, hparams: HyperParameters):
        self.components_layer = LabeledComponents(
            distribution=hparams.distribution,
            components_initializer=hparams.component_initializer,
            labels_initializer=LabelsInitializer(),
        )

    @property
    def prototypes(self):
        return self.components_layer.components.detach().cpu()

    @property
    def prototype_labels(self):
        return self.components_layer.labels.detach().cpu()


class WTACompetitionMixin(BaseYArchitecture):

    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        pass

    def init_inference(self, hparams: HyperParameters):
        self.competition_layer = WTAC()

    def inference(self, comparison_measures, components):
        comp_labels = components[1]
        return self.competition_layer(comparison_measures, comp_labels)


class GLVQLossMixin(BaseYArchitecture):

    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        margin: float = 0.0

        transfer_fn: str = "sigmoid_beta"
        transfer_args: dict = field(default_factory=lambda: dict(beta=10.0))

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


class SingleLearningRateMixin(BaseYArchitecture):

    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        # Training Hyperparameters
        lr: float = 0.01
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam

    def __init__(self, hparams: HyperParameters) -> None:
        super().__init__(hparams)
        self.lr = hparams.lr
        self.optimizer = hparams.optimizer

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)  # type: ignore


class SimpleComparisonMixin(BaseYArchitecture):

    @dataclass
    class HyperParameters(BaseYArchitecture.HyperParameters):
        # Training Hyperparameters
        comparison_fn: Callable = euclidean_distance
        comparison_args: dict = field(default_factory=lambda: dict())

    def init_comparison(self, hparams: HyperParameters):
        self.comparison_layer = LambdaLayer(fn=hparams.comparison_fn,
                                            **hparams.comparison_args)

    def comparison(self, batch, components):
        comp_tensor, _ = components
        batch_tensor, _ = batch

        comp_tensor = comp_tensor.unsqueeze(1)

        distances = self.comparison_layer(batch_tensor, comp_tensor)

        return distances


# ##############################################################################
# GLVQ
# ##############################################################################
class GLVQ(
        SupervisedArchitecture,
        SimpleComparisonMixin,
        GLVQLossMixin,
        WTACompetitionMixin,
        SingleLearningRateMixin,
):
    """GLVQ using the new Scheme
    """

    @dataclass
    class HyperParameters(
            SimpleComparisonMixin.HyperParameters,
            SingleLearningRateMixin.HyperParameters,
            GLVQLossMixin.HyperParameters,
            WTACompetitionMixin.HyperParameters,
            SupervisedArchitecture.HyperParameters,
    ):
        pass
