"""prototorch.models.extras

Modules not yet available in prototorch go here temporarily.

"""

import torch
from prototorch.core.similarities import gaussian


def rank_scaled_gaussian(distances, lambd):
    order = torch.argsort(distances, dim=1)
    ranks = torch.argsort(order, dim=1)
    return torch.exp(-torch.exp(-ranks / lambd) * distances)


def orthogonalization(tensors):
    """Orthogonalization via polar decomposition """
    u, _, v = torch.svd(tensors, compute_uv=True)
    u_shape = tuple(list(u.shape))
    v_shape = tuple(list(v.shape))

    # reshape to (num x N x M)
    u = torch.reshape(u, (-1, u_shape[-2], u_shape[-1]))
    v = torch.reshape(v, (-1, v_shape[-2], v_shape[-1]))

    out = u @ v.permute([0, 2, 1])

    out = torch.reshape(out, u_shape[:-1] + (v_shape[-2], ))

    return out


def ltangent_distance(x, y, omegas):
    r"""Localized Tangent distance.
    Compute Orthogonal Complement: math:`\bm P_k = \bm I - \Omega_k \Omega_k^T`
    Compute Tangent Distance: math:`{\| \bm P \bm x - \bm P_k \bm y_k \|}_2`

    :param `torch.tensor` omegas: Three dimensional matrix
    :rtype: `torch.tensor`
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    p = torch.eye(omegas.shape[-2], device=omegas.device) - torch.bmm(
        omegas, omegas.permute([0, 2, 1]))
    projected_x = x @ p
    projected_y = torch.diagonal(y @ p).T
    expanded_y = torch.unsqueeze(projected_y, dim=1)
    batchwise_difference = expanded_y - projected_x
    differences_squared = batchwise_difference**2
    distances = torch.sqrt(torch.sum(differences_squared, dim=2))
    distances = distances.permute(1, 0)
    return distances


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
