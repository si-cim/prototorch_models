"""ProtoTorch KNN model."""

import warnings

from ..core.competitions import KNNC
from ..core.components import LabeledComponents
from ..core.initializers import LiteralCompInitializer, LiteralLabelsInitializer
from ..utils.utils import parse_data_arg
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
        data, targets = parse_data_arg(data)

        # Layers
        self.proto_layer = LabeledComponents(
            distribution=[],
            components_initializer=LiteralCompInitializer(data),
            labels_initializer=LiteralLabelsInitializer(targets))
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
