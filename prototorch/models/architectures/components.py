from dataclasses import dataclass

from prototorch.core.components import LabeledComponents
from prototorch.core.initializers import (
    AbstractComponentsInitializer,
    LabelsInitializer,
    ZerosCompInitializer,
)
from prototorch.models import BaseYArchitecture


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
        if hparams.component_initializer is not None:
            self.components_layer = LabeledComponents(
                distribution=hparams.distribution,
                components_initializer=hparams.component_initializer,
                labels_initializer=LabelsInitializer(),
            )
            proto_shape = self.components_layer.components.shape[1:]
            self.hparams["initialized_proto_shape"] = proto_shape
        else:
            # when restoring a checkpointed model
            self.components_layer = LabeledComponents(
                distribution=hparams.distribution,
                components_initializer=ZerosCompInitializer(
                    self.hparams["initialized_proto_shape"]),
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
