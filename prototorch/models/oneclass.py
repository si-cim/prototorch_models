""" One Class Classifier based on GLVQ framework """

import torch

from .glvq import GLVQ, SiameseGLVQ, GMLVQ
from prototorch.functions.competitions import wtac_thresh
from prototorch.functions.losses import one_class_classifier_loss
from prototorch.modules import LambdaLayer


class ThetaInitializerPerPrototype():
    def __init__(self, num_thetas, theta=0.1):
        self.theta = theta
        self.num_thetas = num_thetas 

    def generate(self, ):
        return torch.full((self.num_thetas,1), self.theta, requires_grad=True)


class OneClassGLVQ(GLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # initialize Theta Boundary
        self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()
        print(self.theta_boundary)
        print(self.proto_layer._components)
        self.loss = LambdaLayer(one_class_classifier_loss)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert

    def predict_from_distances(self, distances):
        with torch.no_grad():
            plabels = self.proto_layer.component_labels
            y_pred = self.wtac(distances, plabels, self.theta_boundary)
        return y_pred

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(out, y, prototype_labels=plabels, theta_boundary=self.theta_boundary)
        batch_loss = self.transfer_layer(mu, beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)
        return out, loss



class OneClassSiameseGLVQ(SiameseGLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        # initialize Theta Boundary
        self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()


class OneClassGMLVQ(GMLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        # initialize Theta Boundary
        self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()


