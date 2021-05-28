"""Probabilistic GLVQ methods"""

import torch
from prototorch.functions.competitions import stratified_sum
from prototorch.functions.transform import gaussian

from .glvq import GLVQ


def likelihood_loss(probabilities, target, prototype_labels):
    uniques = prototype_labels.unique(sorted=True).tolist()
    labels = target.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    target_indices = torch.LongTensor(list(map(key_val.get, labels)))

    whole_probability = probabilities.sum(dim=1)
    correct_probability = probabilities[torch.arange(len(probabilities)),
                                        target_indices]
    wrong_probability = whole_probability - correct_probability

    likelihood = correct_probability / wrong_probability
    log_likelihood = torch.log(likelihood)
    return log_likelihood


def robust_soft_loss(probabilities, target, prototype_labels):
    uniques = prototype_labels.unique(sorted=True).tolist()
    labels = target.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    target_indices = torch.LongTensor(list(map(key_val.get, labels)))

    whole_probability = probabilities.sum(dim=1)
    correct_probability = probabilities[torch.arange(len(probabilities)),
                                        target_indices]

    likelihood = correct_probability / whole_probability
    log_likelihood = torch.log(likelihood)
    return log_likelihood


class ProbabilisticLVQ(GLVQ):
    def __init__(self, hparams, rejection_confidence=1.0, **kwargs):
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
    """Learning Vector Quantization based on Likelihood Ratios
    """
    @property
    def loss_fn(self):
        return likelihood_loss


class RSLVQ(ProbabilisticLVQ):
    """Learning Vector Quantization based on Likelihood Ratios
    """
    @property
    def loss_fn(self):
        return robust_soft_loss


__all__ = ["LikelihoodRatioLVQ", "RSLVQ"]
