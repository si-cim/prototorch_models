import torch
import torchmetrics
from prototorch.core.competitions import CBCC
from prototorch.core.components import ReasoningComponents
from prototorch.core.initializers import RandomReasoningsInitializer
from prototorch.core.losses import MarginLoss
from prototorch.core.similarities import euclidean_similarity
from prototorch.nn.wrappers import LambdaLayer

from .abstract import ImagePrototypesMixin
from .glvq import SiameseGLVQ


class CBC(SiameseGLVQ):
    """Classification-By-Components."""

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, skip_proto_layer=True, **kwargs)

        similarity_fn = kwargs.get("similarity_fn", euclidean_similarity)
        components_initializer = kwargs.get("components_initializer", None)
        reasonings_initializer = kwargs.get("reasonings_initializer",
                                            RandomReasoningsInitializer())
        self.components_layer = ReasoningComponents(
            self.hparams.distribution,
            components_initializer=components_initializer,
            reasonings_initializer=reasonings_initializer,
        )
        self.similarity_layer = LambdaLayer(similarity_fn)
        self.competition_layer = CBCC()

        # Namespace hook
        self.proto_layer = self.components_layer

        self.loss = MarginLoss(self.hparams.margin)

    def forward(self, x):
        components, reasonings = self.components_layer()
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_components = self.backbone(components)
        self.backbone.requires_grad_(True)
        detections = self.similarity_layer(latent_x, latent_components)
        probs = self.competition_layer(detections, reasonings)
        return probs

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        num_classes = self.num_classes
        y_true = torch.nn.functional.one_hot(y.long(), num_classes=num_classes)
        loss = self.loss(y_pred, y_true).mean()
        return y_pred, loss

    def training_step(self, batch, batch_idx):
        y_pred, train_loss = self.shared_step(batch, batch_idx)
        preds = torch.argmax(y_pred, dim=1)
        accuracy = torchmetrics.functional.accuracy(
            preds.int(),
            batch[1].int(),
            "multiclass",
            num_classes=self.num_classes,
        )
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def predict(self, x):
        with torch.no_grad():
            y_pred = self(x)
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred


class ImageCBC(ImagePrototypesMixin, CBC):
    """CBC model that constrains the components to the range [0, 1] by
    clamping after updates.
    """
