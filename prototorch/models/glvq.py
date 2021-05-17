import torch
import torchmetrics
from prototorch.components import LabeledComponents
from prototorch.functions.activations import get_activation
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import (euclidean_distance, omega_distance,
                                            sed)
from prototorch.functions.helper import get_flat
from prototorch.functions.losses import glvq_loss, lvq1_loss, lvq21_loss

from .abstract import (AbstractPrototypeModel, PrototypeImageModel,
                       SiamesePrototypeModel)


class GLVQ(AbstractPrototypeModel):
    """Generalized Learning Vector Quantization."""


from .abstract import AbstractPrototypeModel, PrototypeImageModel


class GLVQ(AbstractPrototypeModel):
    """Generalized Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.distance_fn = kwargs.get("distance_fn", euclidean_distance)
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam)
        prototype_initializer = kwargs.get("prototype_initializer", None)

        # Default Values
        self.hparams.setdefault("transfer_function", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)

        self.proto_layer = LabeledComponents(
            distribution=self.hparams.distribution,
            initializer=prototype_initializer)

        self.transfer_function = get_activation(self.hparams.transfer_function)
        self.train_acc = torchmetrics.Accuracy()

        self.loss = glvq_loss

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach().cpu()

    def forward(self, x):
        protos, _ = self.proto_layer()
        dis = self.distance_fn(x, protos)
        return dis

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        x, y = train_batch
        dis = self(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(dis, y, prototype_labels=plabels)
        batch_loss = self.transfer_function(mu,
                                            beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)

        # Compute training accuracy
        with torch.no_grad():
            preds = wtac(dis, plabels)

        self.train_acc(preds.int(), y.int())
        # `.int()` because FloatTensors are assumed to be class probabilities

        # Logging
        self.log("train_loss", loss)
        self.log("acc",
                 self.train_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def predict(self, x):
        # model.eval()  # ?!
        with torch.no_grad():
            d = self(x)
            plabels = self.proto_layer.component_labels
            y_pred = wtac(d, plabels)
        return y_pred


class SiameseGLVQ(SiamesePrototypeModel, GLVQ):
    """GLVQ in a Siamese setting.

    GLVQ model that applies an arbitrary transformation on the inputs and the
    prototypes before computing the distances between them. The weights in the
    transformation pipeline are only learned from the inputs.

    """
    def __init__(self,
                 hparams,
                 backbone=torch.nn.Identity(),
                 both_path_gradients=False,
                 **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = backbone
        self.both_path_gradients = both_path_gradients
        self.distance_fn = kwargs.get("distance_fn", sed)

    def forward(self, x):
        protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_protos = self.backbone(protos)
        self.backbone.requires_grad_(True)
        dis = self.distance_fn(latent_x, latent_protos)
        return dis


class GRLVQ(SiamesePrototypeModel, GLVQ):
    """Generalized Relevance Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.relevances = torch.nn.parameter.Parameter(
            torch.ones(self.hparams.input_dim))
        self.distance_fn = kwargs.get("distance_fn", sed)

    @property
    def relevance_profile(self):
        return self.relevances.detach().cpu()

    def backbone(self, x):
        """Namespace hook for the visualization callbacks to work."""
        return x @ torch.diag(self.relevances)

    def forward(self, x):
        protos, _ = self.proto_layer()
        dis = omega_distance(x, protos, torch.diag(self.relevances))
        return dis


class GMLVQ(SiamesePrototypeModel, GLVQ):
    """Generalized Matrix Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = torch.nn.Linear(self.hparams.input_dim,
                                        self.hparams.latent_dim,
                                        bias=False)
        self.distance_fn = kwargs.get("distance_fn", sed)

    @property
    def omega_matrix(self):
        return self.backbone.weight.detach().cpu()

    @property
    def lambda_matrix(self):
        omega = self.backbone.weight  # (latent_dim, input_dim)
        lam = omega.T @ omega
        return lam.detach().cpu()

    def show_lambda(self):
        import matplotlib.pyplot as plt
        title = "Lambda matrix"
        plt.figure(title)
        plt.title(title)
        plt.imshow(self.lambda_matrix, cmap="gray")
        plt.axis("off")
        plt.colorbar()
        plt.show(block=True)

    def forward(self, x):
        protos, _ = self.proto_layer()
        x, protos = get_flat(x, protos)
        latent_x = self.backbone(x)
        latent_protos = self.backbone(protos)
        dis = self.distance_fn(latent_x, latent_protos)
        return dis


class LVQMLN(SiamesePrototypeModel, GLVQ):
    """Learning Vector Quantization Multi-Layer Network.

    GLVQ model that applies an arbitrary transformation on the inputs, BUT NOT
    on the prototypes before computing the distances between them. This of
    course, means that the prototypes no longer live the input space, but
    rather in the embedding space.

    """
    def __init__(self, hparams, backbone=torch.nn.Identity(), **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = backbone

        self.distance_fn = kwargs.get("distance_fn", sed)

    def forward(self, x):
        latent_protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        dis = self.distance_fn(latent_x, latent_protos)
        return dis


class LVQ1(GLVQ):
    """Learning Vector Quantization 1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq1_loss
        self.optimizer = torch.optim.SGD


class LVQ21(GLVQ):
    """Learning Vector Quantization 2.1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq21_loss
        self.optimizer = torch.optim.SGD


class ImageGLVQ(PrototypeImageModel, GLVQ):
    """GLVQ for training on image data.

    GLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """
    pass


class ImageGMLVQ(PrototypeImageModel, GMLVQ):
    """GMLVQ for training on image data.

    GMLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """
    pass
