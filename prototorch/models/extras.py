"""prototorch.models.extras

Modules not yet available in prototorch go here temporarily.

"""

import torch

from ..core.similarities import gaussian


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
