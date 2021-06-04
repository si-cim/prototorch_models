"""ProtoTorch KNN model."""

import warnings

from prototorch.components import LabeledComponents
from prototorch.modules import KNNC

from .abstract import SupervisedPrototypeModel


class KNN(SupervisedPrototypeModel):
    """K-Nearest-Neighbors classification algorithm."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Default hparams
        self.hparams.setdefault("k", 1)

        data = kwargs.get("data", None)
        if data is None:
            raise ValueError("KNN requires data, but was not provided!")

        # Layers
        self.proto_layer = LabeledComponents(initialized_components=data)
        self.competition_layer = KNNC(k=self.hparams.k)

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        return 1  # skip training step

    def on_train_batch_start(self,
                             train_batch,
                             batch_idx,
                             dataloader_idx=None):
        warnings.warn("k-NN has no training, skipping!")
        return -1

    def configure_optimizers(self):
        return None
