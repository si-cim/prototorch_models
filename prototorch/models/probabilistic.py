"""Probabilistic GLVQ methods"""

import torch

from .glvq import GLVQ


# HELPER
# TODO: Refactor into general files, if useful
def probability(distance, variance):
    return torch.exp(-(distance * distance) / (2 * variance))


def grouped_sum(value: torch.Tensor,
                labels: torch.LongTensor) -> (torch.Tensor, torch.LongTensor):
    """Group-wise average for (sparse) grouped tensors

    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)

    Returns:
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)

    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2,  0.2,  0.2 ],    #-> group / class 3
                             [0.4,  0.4,  0.4 ],    #-> group / class 3
                             [0.0,  0.0,  0.0 ]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)

        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
                [0.1500, 0.1500, 0.1500],
                [0.3000, 0.3000, 0.3000]])

        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique(sorted=True).tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels = labels.unique(dim=0)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
        0, labels, value)
    return result.T


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


class LikelihoodRatioLVQ(GLVQ):
    """Learning Vector Quantization based on Likelihood Ratios

    Based on "Soft Learning Vector Quantization" from Sambu Seo and Klaus Obermayer (2003).
    """
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = probability

    def forward(self, x):
        distances = self._forward(x)
        conditional = self.conditional_distribution(distances,
                                                    self.hparams.variance)
        prior = 1.0 / torch.Tensor(self.proto_layer.distribution).sum().item()
        posterior = conditional * prior

        plabels = torch.LongTensor(self.proto_layer.component_labels)
        y_pred = grouped_sum(posterior.T, plabels)

        return y_pred

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        X, y = batch
        out = self.forward(X)
        plabels = self.proto_layer.component_labels
        batch_loss = -likelihood_loss(out, y, prototype_labels=plabels)
        loss = batch_loss.sum(dim=0)

        return loss

    def predict(self, x):
        probabilities = self.forward(x)
        confidence, prediction = torch.max(probabilities, dim=1)
        prediction[confidence < 0.1] = -1
        return prediction


class RSLVQ(GLVQ):
    """Learning Vector Quantization based on Likelihood Ratios

    Based on "Soft Learning Vector Quantization" from Sambu Seo and Klaus Obermayer (2003).
    """
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        self.conditional_distribution = probability

    def forward(self, x):
        distances = self._forward(x)
        conditional = self.conditional_distribution(distances,
                                                    self.hparams.variance)
        prior = 1.0 / torch.Tensor(self.proto_layer.distribution).sum().item()
        posterior = conditional * prior

        plabels = torch.LongTensor(self.proto_layer.component_labels)
        y_pred = grouped_sum(posterior.T, plabels)

        return y_pred

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        X, y = batch
        out = self.forward(X)
        plabels = self.proto_layer.component_labels
        batch_loss = -robust_soft_loss(out, y, prototype_labels=plabels)
        loss = batch_loss.sum(dim=0)

        return loss

    def predict(self, x):
        probabilities = self.forward(x)
        confidence, prediction = torch.max(probabilities, dim=1)
        #prediction[confidence < 0.1] = -1
        return prediction


__all__ = ["LikelihoodRatioLVQ", "probability", "grouped_sum"]
