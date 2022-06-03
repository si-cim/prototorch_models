from dataclasses import dataclass

from prototorch.core.components import LabeledComponents
from prototorch.core.initializers import (
    AbstractComponentsInitializer,
    LabelsInitializer,
)
from prototorch.y import BaseYArchitecture


class SupervisedArchitecture(BaseYArchitecture):
    """
    Supervised Architecture

    An architecture that uses labeled Components as component Layer.
    """
    components_layer: LabeledComponents

    # HyperParameters
    # ----------------------------------------------------------------------------------------------------
    @dataclass
    class HyperParameters:
        """
        distribution: A valid prototype distribution. No default possible.
        components_initializer: An implementation of AbstractComponentsInitializer. No default possible.
        """
        distribution: "dict[str, int]"
        component_initializer: AbstractComponentsInitializer

    # Steps
    # ----------------------------------------------------------------------------------------------------
    def init_components(self, hparams: HyperParameters):
        self.components_layer = LabeledComponents(
            distribution=hparams.distribution,
            components_initializer=hparams.component_initializer,
            labels_initializer=LabelsInitializer(),
        )

    # Properties
    # ----------------------------------------------------------------------------------------------------
    @property
    def prototypes(self):
        """
        Returns the position of the prototypes.
        """
        return self.components_layer.components.detach().cpu()

    @property
    def prototype_labels(self):
        """
        Returns the labels of the prototypes.
        """
        return self.components_layer.labels.detach().cpu()
