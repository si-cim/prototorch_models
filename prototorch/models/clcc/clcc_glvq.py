from dataclasses import dataclass
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
from prototorch.models.clcc.clcc_scheme import CLCCScheme
from prototorch.nn.wrappers import LambdaLayer


class SupervisedScheme(CLCCScheme):

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


# ##############################################################################
# GLVQ
# ##############################################################################
class GLVQ(
        SupervisedScheme, ):
    """GLVQ using the new Scheme
    """

    @dataclass
    class HyperParameters(SupervisedScheme.HyperParameters):
        distance_fn: Callable = euclidean_distance
        lr: float = 0.01
        margin: float = 0.0
        # TODO: make nicer
        transfer_fn: str = "identity"
        transfer_beta: float = 10.0
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam

    def __init__(self, hparams: HyperParameters) -> None:
        super().__init__(hparams)
        self.lr = hparams.lr
        self.optimizer = hparams.optimizer

    def init_comparison(self, hparams: HyperParameters):
        self.comparison_layer = LambdaLayer(hparams.distance_fn)

    def init_inference(self, hparams: HyperParameters):
        self.competition_layer = WTAC()

    def init_loss(self, hparams):
        self.loss_layer = GLVQLoss(
            margin=hparams.margin,
            transfer_fn=hparams.transfer_fn,
            beta=hparams.transfer_beta,
        )

    # Steps
    def comparison(self, batch, components):
        comp_tensor, _ = components
        batch_tensor, _ = batch

        comp_tensor = comp_tensor.unsqueeze(1)

        distances = self.comparison_layer(batch_tensor, comp_tensor)

        return distances

    def inference(self, comparisonmeasures, components):
        comp_labels = components[1]
        return self.competition_layer(comparisonmeasures, comp_labels)

    def loss(self, comparisonmeasures, batch, components):
        target = batch[1]
        comp_labels = components[1]
        return self.loss_layer(comparisonmeasures, target, comp_labels)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)  # type: ignore

    # Properties
    @property
    def prototypes(self):
        return self.components_layer.components.detach().cpu()

    @property
    def prototype_labels(self):
        return self.components_layer.labels.detach().cpu()
