import torch
import torchmetrics
from prototorch.components import LabeledComponents
from prototorch.functions.activations import get_activation
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import (euclidean_distance, omega_distance,
                                            squared_euclidean_distance)
from prototorch.functions.losses import glvq_loss, lvq1_loss, lvq21_loss

from .abstract import AbstractPrototypeModel

from torch.optim.lr_scheduler import ExponentialLR


class GLVQ(AbstractPrototypeModel):
    """Generalized Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        # Default Values
        self.hparams.setdefault("distance", euclidean_distance)
        self.hparams.setdefault("optimizer", torch.optim.Adam)
        self.hparams.setdefault("transfer_function", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)

        self.proto_layer = LabeledComponents(
            labels=(self.hparams.nclasses, self.hparams.prototypes_per_class),
            initializer=self.hparams.prototype_initializer)

        self.transfer_function = get_activation(self.hparams.transfer_function)
        self.train_acc = torchmetrics.Accuracy()

        self.loss = glvq_loss

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach().cpu()

    def forward(self, x):
        protos, _ = self.proto_layer()
        dis = self.hparams.distance(x, protos)
        return dis

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        x, y = train_batch
        x = x.view(x.size(0), -1)  # flatten
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
        return y_pred.numpy()


class LVQ1(GLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq1_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer,
                                  gamma=0.99,
                                  last_epoch=-1,
                                  verbose=False)
        sch = {
            "scheduler": scheduler,
            "interval": "step",
        }  # called after each training step
        return [optimizer], [sch]


class LVQ21(GLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq21_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer,
                                  gamma=0.99,
                                  last_epoch=-1,
                                  verbose=False)
        sch = {
            "scheduler": scheduler,
            "interval": "step",
        }  # called after each training step
        return [optimizer], [sch]


class ImageGLVQ(GLVQ):
    """GLVQ for training on image data.

    GLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.proto_layer.components.data.clamp_(0.0, 1.0)


class SiameseGLVQ(GLVQ):
    """GLVQ in a Siamese setting.

    GLVQ model that applies an arbitrary transformation on the inputs and the
    prototypes before computing the distances between them. The weights in the
    transformation pipeline are only learned from the inputs.

    """
    def __init__(self,
                 hparams,
                 backbone_module=torch.nn.Identity,
                 backbone_params={},
                 sync=True,
                 **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = backbone_module(**backbone_params)
        self.backbone_dependent = backbone_module(
            **backbone_params).requires_grad_(False)
        self.sync = sync

    def sync_backbones(self):
        master_state = self.backbone.state_dict()
        self.backbone_dependent.load_state_dict(master_state, strict=True)

    def configure_optimizers(self):
        optim = self.hparams.optimizer
        proto_opt = optim(self.proto_layer.parameters(),
                          lr=self.hparams.proto_lr)
        if list(self.backbone.parameters()):
            # only add an optimizer is the backbone has trainable parameters
            # otherwise, the next line fails
            bb_opt = optim(self.backbone.parameters(), lr=self.hparams.bb_lr)
            return proto_opt, bb_opt
        else:
            return proto_opt

    def forward(self, x):
        if self.sync:
            self.sync_backbones()
        protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        latent_protos = self.backbone_dependent(protos)
        dis = euclidean_distance(latent_x, latent_protos)
        return dis

    def predict_latent(self, x):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        # model.eval()  # ?!
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            latent_protos = self.backbone_dependent(protos)
            d = euclidean_distance(x, latent_protos)
            y_pred = wtac(d, plabels)
        return y_pred.numpy()


class GRLVQ(GLVQ):
    """Generalized Relevance Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.relevances = torch.nn.parameter.Parameter(
            torch.ones(self.hparams.input_dim))

    def forward(self, x):
        protos, _ = self.proto_layer()
        dis = omega_distance(x, protos, torch.diag(self.relevances))
        return dis

    def backbone(self, x):
        return x @ torch.diag(self.relevances)

    @property
    def relevance_profile(self):
        return self.relevances.detach().cpu()

    def predict_latent(self, x):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        # model.eval()  # ?!
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            latent_protos = protos @ torch.diag(self.relevances)
            d = squared_euclidean_distance(x, latent_protos)
            y_pred = wtac(d, plabels)
        return y_pred.numpy()


class GMLVQ(GLVQ):
    """Generalized Matrix Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.omega_layer = torch.nn.Linear(self.hparams.input_dim,
                                           self.hparams.latent_dim,
                                           bias=False)

        # Namespace hook for the visualization callbacks to work
        self.backbone = self.omega_layer

    @property
    def omega_matrix(self):
        return self.omega_layer.weight.detach().cpu()

    @property
    def lambda_matrix(self):
        omega = self.omega_layer.weight  # (latent_dim, input_dim)
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
        latent_x = self.omega_layer(x)
        latent_protos = self.omega_layer(protos)
        dis = squared_euclidean_distance(latent_x, latent_protos)
        return dis

    def predict_latent(self, x):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        # model.eval()  # ?!
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            latent_protos = self.omega_layer(protos)
            d = squared_euclidean_distance(x, latent_protos)
            y_pred = wtac(d, plabels)
        return y_pred.numpy()


class LVQMLN(GLVQ):
    """Learning Vector Quantization Multi-Layer Network.

    GLVQ model that applies an arbitrary transformation on the inputs, BUT NOT
    on the prototypes before computing the distances between them. This of
    course, means that the prototypes no longer live the input space, but
    rather in the embedding space.

    """
    def __init__(self,
                 hparams,
                 backbone_module=torch.nn.Identity,
                 backbone_params={},
                 **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = backbone_module(**backbone_params)
        with torch.no_grad():
            protos = self.backbone(self.proto_layer()[0])
        self.proto_layer.load_state_dict({"_components": protos}, strict=False)

    def forward(self, x):
        latent_protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        dis = euclidean_distance(latent_x, latent_protos)
        return dis

    def predict_latent(self, x):
        """Predict `x` assuming it is already embedded in the latent space."""
        with torch.no_grad():
            latent_protos, plabels = self.proto_layer()
            d = euclidean_distance(x, latent_protos)
            y_pred = wtac(d, plabels)
        return y_pred.numpy()
