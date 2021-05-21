import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.components.components import Components
from prototorch.functions.distances import euclidean_distance
from prototorch.functions.similarities import cosine_similarity

from .abstract import AbstractPrototypeModel, PrototypeImageModel
from .glvq import SiameseGLVQ


def rescaled_cosine_similarity(x, y):
    """Cosine Similarity rescaled to [0, 1]."""
    similarities = cosine_similarity(x, y)
    return (similarities + 1.0) / 2.0


def shift_activation(x):
    return (x + 1.0) / 2.0


def euclidean_similarity(x, y, beta=3):
    d = euclidean_distance(x, y)
    return torch.exp(-d * beta)


class CosineSimilarity(torch.nn.Module):
    def __init__(self, activation=shift_activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, y):
        epsilon = torch.finfo(x.dtype).eps
        normed_x = (x / x.pow(2).sum(dim=tuple(range(
            1, x.ndim)), keepdim=True).clamp(min=epsilon).sqrt()).flatten(
                start_dim=1)
        normed_y = (y / y.pow(2).sum(dim=tuple(range(
            1, y.ndim)), keepdim=True).clamp(min=epsilon).sqrt()).flatten(
                start_dim=1)
        # normed_x = (x / torch.linalg.norm(x, dim=1))
        diss = torch.inner(normed_x, normed_y)
        return self.activation(diss)


class MarginLoss(torch.nn.modules.loss._Loss):
    def __init__(self,
                 margin=0.3,
                 size_average=None,
                 reduce=None,
                 reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input_, target):
        dp = torch.sum(target * input_, dim=-1)
        dm = torch.max(input_ - target, dim=-1).values
        return torch.nn.functional.relu(dm - dp + self.margin)


class ReasoningLayer(torch.nn.Module):
    def __init__(self, num_components, num_classes, n_replicas=1):
        super().__init__()
        self.n_replicas = n_replicas
        self.num_classes = num_classes
        probabilities_init = torch.zeros(2, 1, num_components,
                                         self.num_classes)
        probabilities_init.uniform_(0.4, 0.6)
        self.reasoning_probabilities = torch.nn.Parameter(probabilities_init)

    @property
    def reasonings(self):
        pk = self.reasoning_probabilities[0]
        nk = (1 - pk) * self.reasoning_probabilities[1]
        ik = 1 - pk - nk
        img = torch.cat([pk, nk, ik], dim=0).permute(1, 0, 2)
        return img.unsqueeze(1)

    def forward(self, detections):
        pk = self.reasoning_probabilities[0].clamp(0, 1)
        nk = (1 - pk) * self.reasoning_probabilities[1].clamp(0, 1)
        epsilon = torch.finfo(pk.dtype).eps
        numerator = (detections @ (pk - nk)) + nk.sum(1)
        probs = numerator / (pk + nk).sum(1)
        probs = probs.squeeze(0)
        return probs


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
        nclasses = self.reasoning_layer.num_classes
        y_true = torch.nn.functional.one_hot(y.long(), num_classes=nclasses)
        loss = MarginLoss(self.margin)(y_pred, y_true).mean(dim=0)
        return y_pred, loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        y_pred, train_loss = self.shared_step(batch, batch_idx, optimizer_idx)
        preds = torch.argmax(y_pred, dim=1)
        self.acc_metric(preds.int(), batch[1].int())
        self.log("train_acc",
                 self.acc_metric,
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


class ImageCBC(CBC):
    """CBC model that constrains the components to the range [0, 1] by
    clamping after updates.
    """
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self.component_layer.components.data.clamp_(0.0, 1.0)
