"""Probabilistic GLVQ methods"""

import torch
from prototorch.functions.competitions import stratified_sum
from prototorch.functions.losses import log_likelihood_ratio_loss, robust_soft_loss
from prototorch.functions.transforms import gaussian

from .glvq import GLVQ


class ProbabilisticLVQ(GLVQ):
    def __init__(self, hparams, rejection_confidence=0.0, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = gaussian
        self.rejection_confidence = rejection_confidence

    def forward(self, x):
        distances = self._forward(x)
        conditional = self.conditional_distribution(distances,
                                                    self.hparams.variance)
        prior = (1. / self.num_prototypes) * torch.ones(self.num_prototypes)
        posterior = conditional * prior
        plabels = self.proto_layer._labels
        y_pred = stratified_sum(posterior, plabels)
        return y_pred

    def predict(self, x):
        y_pred = self.forward(x)
        confidence, prediction = torch.max(y_pred, dim=1)
        prediction[confidence < self.rejection_confidence] = -1
        return prediction

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        X, y = batch
        out = self.forward(X)
        plabels = self.proto_layer.component_labels
        batch_loss = -self.loss_fn(out, y, plabels)
        loss = batch_loss.sum(dim=0)

        return loss


class LikelihoodRatioLVQ(ProbabilisticLVQ):
    """Learning Vector Quantization based on Likelihood Ratios."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = log_likelihood_ratio_loss


class RSLVQ(ProbabilisticLVQ):
    """Robust Soft Learning Vector Quantization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = robust_soft_loss
