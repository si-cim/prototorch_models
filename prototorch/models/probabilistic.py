"""Probabilistic GLVQ methods"""

import torch
from prototorch.functions.losses import nllr_loss, rslvq_loss
from prototorch.functions.pooling import stratified_min_pooling, stratified_sum_pooling
from prototorch.functions.transforms import gaussian
from prototorch.modules import LambdaLayer, LossLayer

from .glvq import GLVQ


class CELVQ(GLVQ):
    """Cross-Entropy Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Loss
        self.loss = torch.nn.CrossEntropyLoss()

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)  # [None, num_protos]
        plabels = self.proto_layer.component_labels
        winning = stratified_min_pooling(out, plabels)  # [None, num_classes]
        probs = -1.0 * winning
        batch_loss = self.loss(probs, y.long())
        loss = batch_loss.sum(dim=0)
        return out, loss


class ProbabilisticLVQ(GLVQ):
    def __init__(self, hparams, rejection_confidence=0.0, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = gaussian
        self.rejection_confidence = rejection_confidence

    def forward(self, x):
        distances = self.compute_distances(x)
        conditional = self.conditional_distribution(distances,
                                                    self.hparams.variance)
        prior = (1. / self.num_prototypes) * torch.ones(self.num_prototypes,
                                                        device=self.device)
        posterior = conditional * prior
        plabels = self.proto_layer._labels
        y_pred = stratified_sum_pooling(posterior, plabels)
        return y_pred

    def predict(self, x):
        y_pred = self.forward(x)
        confidence, prediction = torch.max(y_pred, dim=1)
        prediction[confidence < self.rejection_confidence] = -1
        return prediction

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.forward(x)
        plabels = self.proto_layer.component_labels
        batch_loss = self.loss(out, y, plabels)
        loss = batch_loss.sum(dim=0)
        return loss


class SLVQ(ProbabilisticLVQ):
    """Soft Learning Vector Quantization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = LossLayer(nllr_loss)


class RSLVQ(ProbabilisticLVQ):
    """Robust Soft Learning Vector Quantization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = LossLayer(rslvq_loss)
