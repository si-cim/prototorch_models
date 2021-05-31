"""Probabilistic GLVQ methods"""

import torch
from prototorch.functions.competitions import stratified_sum
from prototorch.functions.losses import log_likelihood_ratio_loss, robust_soft_loss
from prototorch.functions.transform import gaussian

from .glvq import GLVQ


class ProbabilisticLVQ(GLVQ):
    def __init__(self, hparams, rejection_confidence=0.0, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = gaussian
        self.rejection_confidence = rejection_confidence

    def predict(self, x):
        probabilities = self.forward(x)
        confidence, prediction = torch.max(probabilities, dim=1)
        prediction[confidence < self.rejection_confidence] = -1
        return prediction

    def forward(self, x):
        distances = self._forward(x)
        conditional = self.conditional_distribution(distances,
                                                    self.hparams.variance)
        prior = 1.0 / torch.Tensor(self.proto_layer.distribution).sum().item()
        posterior = conditional * prior

        plabels = torch.LongTensor(self.proto_layer.component_labels)
        y_pred = stratified_sum(posterior.T, plabels)

        return y_pred

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
