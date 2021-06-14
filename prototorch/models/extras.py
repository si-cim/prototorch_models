"""prototorch.models.extras

Modules not yet available in prototorch go here temporarily.

"""

import torch

from ..core.distances import euclidean_distance
from ..core.similarities import cosine_similarity


def rescaled_cosine_similarity(x, y):
    """Cosine Similarity rescaled to [0, 1]."""
    similarities = cosine_similarity(x, y)
    return (similarities + 1.0) / 2.0


def shift_activation(x):
    return (x + 1.0) / 2.0


def euclidean_similarity(x, y, variance=1.0):
    d = euclidean_distance(x, y)
    return torch.exp(-(d * d) / (2 * variance))


def gaussian(distances, variance):
    return torch.exp(-(distances * distances) / (2 * variance))


def rank_scaled_gaussian(distances, lambd):
    order = torch.argsort(distances, dim=1)
    ranks = torch.argsort(order, dim=1)

    return torch.exp(-torch.exp(-ranks / lambd) * distances)


class GaussianPrior(torch.nn.Module):
    def __init__(self, variance):
        super().__init__()
        self.variance = variance

    def forward(self, distances):
        return gaussian(distances, self.variance)


class RankScaledGaussianPrior(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, distances):
        return rank_scaled_gaussian(distances, self.lambd)


class ConnectionTopology(torch.nn.Module):
    def __init__(self, agelimit, num_prototypes):
        super().__init__()
        self.agelimit = agelimit
        self.num_prototypes = num_prototypes

        self.cmat = torch.zeros((self.num_prototypes, self.num_prototypes))
        self.age = torch.zeros_like(self.cmat)

    def forward(self, d):
        order = torch.argsort(d, dim=1)

        for element in order:
            i0, i1 = element[0], element[1]

            self.cmat[i0][i1] = 1
            self.cmat[i1][i0] = 1

            self.age[i0][i1] = 0
            self.age[i1][i0] = 0

            self.age[i0][self.cmat[i0] == 1] += 1
            self.age[i1][self.cmat[i1] == 1] += 1

            self.cmat[i0][self.age[i0] > self.agelimit] = 0
            self.cmat[i1][self.age[i1] > self.agelimit] = 0

    def get_neighbors(self, position):
        return torch.where(self.cmat[position])

    def add_prototype(self):
        new_cmat = torch.zeros([dim + 1 for dim in self.cmat.shape])
        new_cmat[:-1, :-1] = self.cmat
        self.cmat = new_cmat

        new_age = torch.zeros([dim + 1 for dim in self.age.shape])
        new_age[:-1, :-1] = self.age
        self.age = new_age

    def add_connection(self, a, b):
        self.cmat[a][b] = 1
        self.cmat[b][a] = 1

        self.age[a][b] = 0
        self.age[b][a] = 0

    def remove_connection(self, a, b):
        self.cmat[a][b] = 0
        self.cmat[b][a] = 0

        self.age[a][b] = 0
        self.age[b][a] = 0

    def extra_repr(self):
        return f"(agelimit): ({self.agelimit})"


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
    def __init__(self, num_components, num_classes, num_replicas=1):
        super().__init__()
        self.num_replicas = num_replicas
        self.num_classes = num_classes
        probabilities_init = torch.zeros(2, 1, num_components,
                                         self.num_classes)
        probabilities_init.uniform_(0.4, 0.6)
        # TODO Use `self.register_parameter("param", Paramater(param))` instead
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
        numerator = (detections @ (pk - nk)) + nk.sum(1)
        probs = numerator / (pk + nk).sum(1)
        probs = probs.squeeze(0)
        return probs
