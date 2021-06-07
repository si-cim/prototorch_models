""" One Class Classifier based on GLVQ framework """

import torch
from torch.nn.parameter import Parameter

from .glvq import GLVQ, SiameseGLVQ, GMLVQ
from prototorch.functions.competitions import wtac_thresh
from prototorch.functions.losses import one_class_classifier_loss, one_class_classifier_triplet_loss
from prototorch.modules import LambdaLayer

from prototorch.functions.distances import (
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
    euclidean_distance,
)


class ThetaInitializerPerPrototype():
    def __init__(self, num_thetas, theta=0.02):
        self.theta = theta
        self.num_thetas = num_thetas 

    def generate(self, ):
        return torch.full((self.num_thetas,1), self.theta, requires_grad=True)


class OneClassGLVQ(GLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)

        # Additional parameters
        #self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()
        theta = torch.randn(self.proto_layer.component_labels.shape,
                            device=self.device)
        theta = torch.pow(theta, 2)
        self.register_parameter("_theta", Parameter(theta))

        self.loss = LambdaLayer(one_class_classifier_loss)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert

    def theta_boundary(self):
        return self._theta.detach().cpu()

    def predict_from_distances(self, distances):
        with torch.no_grad():
            plabels = self.proto_layer.component_labels
            y_pred = self.wtac(distances, plabels, self._theta)
        return y_pred

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(out, y, prototype_labels=plabels, theta_boundary=self._theta)
        batch_loss = self.transfer_layer(mu, beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)
        return out, loss


class OneClassGMLVQ(GMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)

        # Additional parameters
        #self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()
        theta = torch.randn(self.proto_layer.component_labels.shape,
                            device=self.device)
        theta = torch.pow(theta, 2)
        self.register_parameter("_theta", Parameter(theta))

        self.loss = LambdaLayer(one_class_classifier_loss)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert

    def theta_boundary(self):
        return self._theta.detach().cpu()

    def predict_from_distances(self, distances):
        with torch.no_grad():
            plabels = self.proto_layer.component_labels
            y_pred = self.wtac(distances, plabels, self._theta)
        return y_pred

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(out, y, prototype_labels=plabels, theta_boundary=self._theta)
        batch_loss = self.transfer_layer(mu, beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)
        return out, loss

