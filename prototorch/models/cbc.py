import torch
import torchmetrics

from .abstract import ImagePrototypesMixin
from .extras import (
    CosineSimilarity,
    MarginLoss,
    ReasoningLayer,
    euclidean_similarity,
    rescaled_cosine_similarity,
    shift_activation,
)
from .glvq import SiameseGLVQ


class CBC(SiameseGLVQ):
    """Classification-By-Components."""
    def __init__(self,
                 hparams,
                 margin=0.1,
                 similarity=euclidean_similarity,
                 **kwargs):
        super().__init__(hparams, **kwargs)
        self.margin = margin
        self.similarity_fn = kwargs.get("similarity_fn", euclidean_similarity)
        num_components = self.components.shape[0]
        self.reasoning_layer = ReasoningLayer(num_components=num_components,
                                              num_classes=self.num_classes)
        self.component_layer = self.proto_layer

    @property
    def components(self):
        return self.prototypes

    @property
    def reasonings(self):
        return self.reasoning_layer.reasonings.cpu()

    def forward(self, x):
        components, _ = self.component_layer()
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_components = self.backbone(components)
        self.backbone.requires_grad_(True)
        detections = self.similarity_fn(latent_x, latent_components)
        probs = self.reasoning_layer(detections)
        return probs

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        # x = x.view(x.size(0), -1)
        y_pred = self(x)
        num_classes = self.reasoning_layer.num_classes
        y_true = torch.nn.functional.one_hot(y.long(), num_classes=num_classes)
        loss = MarginLoss(self.margin)(y_pred, y_true).mean(dim=0)
        return y_pred, loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        y_pred, train_loss = self.shared_step(batch, batch_idx, optimizer_idx)
        preds = torch.argmax(y_pred, dim=1)
        accuracy = torchmetrics.functional.accuracy(preds.int(),
                                                    batch[1].int())
        self.log("train_acc",
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
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
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        # Namespace hook
        self.proto_layer = self.component_layer
