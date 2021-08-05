"""Probabilistic GLVQ methods"""

import torch

from ..core.losses import nllr_loss, rslvq_loss
from ..core.pooling import stratified_min_pooling, stratified_sum_pooling
from ..nn.wrappers import LambdaLayer, LossLayer
from .extras import GaussianPrior, RankScaledGaussianPrior
from .glvq import GLVQ, SiameseGMLVQ


class CELVQ(GLVQ):
    """Cross-Entropy Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Loss
        self.loss = torch.nn.CrossEntropyLoss()

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)  # [None, num_protos]
        _, plabels = self.proto_layer()
        winning = stratified_min_pooling(out, plabels)  # [None, num_classes]
        probs = -1.0 * winning
        batch_loss = self.loss(probs, y.long())
        loss = batch_loss.sum()
        return out, loss


class ProbabilisticLVQ(GLVQ):
    def __init__(self, hparams, rejection_confidence=0.0, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = None
        self.rejection_confidence = rejection_confidence

    def forward(self, x):
        distances = self.compute_distances(x)
        conditional = self.conditional_distribution(distances)
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
        _, plabels = self.proto_layer()
        batch_loss = self.loss(out, y, plabels)
        loss = batch_loss.sum()
        return loss


class SLVQ(ProbabilisticLVQ):
    """Soft Learning Vector Quantization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = LossLayer(nllr_loss)
        self.conditional_distribution = GaussianPrior(self.hparams.variance)


class RSLVQ(ProbabilisticLVQ):
    """Robust Soft Learning Vector Quantization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = LossLayer(rslvq_loss)
        self.conditional_distribution = GaussianPrior(self.hparams.variance)


class PLVQ(ProbabilisticLVQ, SiameseGMLVQ):
    """Probabilistic Learning Vector Quantization.

    TODO: Use Backbone LVQ instead
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional_distribution = RankScaledGaussianPrior(
            self.hparams.lambd)
        self.loss = torch.nn.KLDivLoss()

    # FIXME
    # def training_step(self, batch, batch_idx, optimizer_idx=None):
    #     x, y = batch
    #     y_pred = self(x)
    #     batch_loss = self.loss(y_pred, y)
    #     loss = batch_loss.sum()
    #     return loss
