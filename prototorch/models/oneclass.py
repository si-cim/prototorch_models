""" One Class Classifier based on GLVQ framework """

from .glvq import GLVQ, SiameseGLVQ, GMLVQ



class ThetaInitializerPerPrototype():
    def __init__(self, num_thetas, theta=[0.1]):
        self.num_thetas = num_thetas 

    def generate(self, ):
        return torch.arange(self.theta).repeat(self.num_theta, 1)




class OneClassGLVQ(GLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        # initialize Theta Boundary
        self.theta_boundary = ThetaInitializerPerPrototype(num_thetas=len(self.prototype_labels)).generate()


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


