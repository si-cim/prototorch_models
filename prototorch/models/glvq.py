"""Models based on the GLVQ framework."""

import torch
import torchmetrics
from prototorch.components import LabeledComponents
from prototorch.functions.activations import get_activation
from prototorch.functions.competitions import stratified_min, wtac
from prototorch.functions.distances import (euclidean_distance, omega_distance,
                                            sed)
from prototorch.functions.helper import get_flat
from prototorch.functions.losses import glvq_loss, lvq1_loss, lvq21_loss

from .abstract import AbstractPrototypeModel, PrototypeImageModel


class GLVQ(AbstractPrototypeModel):
    """Generalized Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.distance_fn = kwargs.get("distance_fn", euclidean_distance)
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam)

        # Default Values
        self.hparams.setdefault("transfer_fn", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)
        self.hparams.setdefault("lr", 0.01)

        self.proto_layer = LabeledComponents(
            distribution=self.hparams.distribution,
            initializer=self.prototype_initializer(**kwargs))

        self.transfer_fn = get_activation(self.hparams.transfer_fn)
        self.acc_metric = torchmetrics.Accuracy()

        self.loss = glvq_loss

    def prototype_initializer(self, **kwargs):
        return kwargs.get("prototype_initializer", None)

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach().cpu()

    @property
    def num_classes(self):
        return len(self.proto_layer.distribution)

    def _forward(self, x):
        protos, _ = self.proto_layer()
        distances = self.distance_fn(x, protos)
        return distances

    def forward(self, x):
        distances = self._forward(x)
        y_pred = self.predict_from_distances(distances)
        y_pred = torch.eye(self.num_classes, device=self.device)[y_pred.int()]
        return y_pred

    def predict_from_distances(self, distances):
        with torch.no_grad():
            plabels = self.proto_layer.component_labels
            y_pred = wtac(distances, plabels)
        return y_pred

    def predict(self, x):
        with torch.no_grad():
            distances = self._forward(x)
        y_pred = self.predict_from_distances(distances)
        return y_pred

    def log_acc(self, distances, targets, tag):
        preds = self.predict_from_distances(distances)
        self.acc_metric(preds.int(), targets.int())
        # `.int()` because FloatTensors are assumed to be class probabilities

        self.log(tag,
                 self.acc_metric,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self._forward(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(out, y, prototype_labels=plabels)
        batch_loss = self.transfer_fn(mu, beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)
        return out, loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        out, train_loss = self.shared_step(batch, batch_idx, optimizer_idx)
        self.log("train_loss", train_loss)
        self.log_acc(out, batch[-1], tag="train_acc")
        return train_loss

    def validation_step(self, batch, batch_idx):
        # `model.eval()` and `torch.no_grad()` handled by pl
        out, val_loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", val_loss)
        self.log_acc(out, batch[-1], tag="val_acc")
        return val_loss

    def test_step(self, batch, batch_idx):
        # `model.eval()` and `torch.no_grad()` handled by pl
        out, test_loss = self.shared_step(batch, batch_idx)
        self.log_acc(out, batch[-1], tag="test_acc")
        return test_loss

    def test_epoch_end(self, outputs):
        test_loss = 0.0
        for batch_loss in outputs:
            test_loss += batch_loss.item()
        self.log("test_loss", test_loss)

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     pass

    def increase_prototypes(self, initializer, distribution):
        self.proto_layer.increase_components(initializer, distribution)

    def __repr__(self):
        super_repr = super().__repr__()
        return f"{super_repr}"


class SiameseGLVQ(GLVQ):
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

    def configure_optimizers(self):
        proto_opt = self.optimizer(self.proto_layer.parameters(),
                                   lr=self.hparams.proto_lr)
        if list(self.backbone.parameters()):
            # only add an optimizer is the backbone has trainable parameters
            # otherwise, the next line fails
            bb_opt = self.optimizer(self.backbone.parameters(),
                                    lr=self.hparams.bb_lr)
            return proto_opt, bb_opt
        else:
            return proto_opt

    def _forward(self, x):
        protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_protos = self.backbone(protos)
        self.backbone.requires_grad_(True)
        distances = self.distance_fn(latent_x, latent_protos)
        return distances

    def predict_latent(self, x, map_protos=True):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        self.eval()
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            if map_protos:
                protos = self.backbone(protos)
            d = self.distance_fn(x, protos)
            y_pred = wtac(d, plabels)
        return y_pred


class GRLVQ(SiameseGLVQ):
    """Generalized Relevance Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.relevances = torch.nn.parameter.Parameter(
            torch.ones(self.hparams.input_dim))

        # Overwrite backbone
        self.backbone = self._backbone

    @property
    def relevance_profile(self):
        return self.relevances.detach().cpu()

    def _backbone(self, x):
        """Namespace hook for the visualization callbacks to work."""
        return x @ torch.diag(self.relevances)

    def _forward(self, x):
        protos, _ = self.proto_layer()
        distances = omega_distance(x, protos, torch.diag(self.relevances))
        return distances


class GMLVQ(SiameseGLVQ):
    """Generalized Matrix Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = torch.nn.Linear(self.hparams.input_dim,
                                        self.hparams.latent_dim,
                                        bias=False)

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

    def _forward(self, x):
        protos, _ = self.proto_layer()
        x, protos = get_flat(x, protos)
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_protos = self.backbone(protos)
        self.backbone.requires_grad_(True)
        distances = self.distance_fn(latent_x, latent_protos)
        return distances


class LVQMLN(SiameseGLVQ):
    """Learning Vector Quantization Multi-Layer Network.

    GLVQ model that applies an arbitrary transformation on the inputs, BUT NOT
    on the prototypes before computing the distances between them. This of
    course, means that the prototypes no longer live the input space, but
    rather in the embedding space.

    """
    def _forward(self, x):
        latent_protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        distances = self.distance_fn(latent_x, latent_protos)
        return distances


class CELVQ(GLVQ):
    """Cross-Entropy Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss()

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self._forward(x)  # [None, num_protos]
        plabels = self.proto_layer.component_labels
        probs = -1.0 * stratified_min(out, plabels)  # [None, num_classes]
        batch_loss = self.loss(out, y.long())
        loss = batch_loss.sum(dim=0)
        return out, loss


class GLVQ1(GLVQ):
    """Generalized Learning Vector Quantization 1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq1_loss
        self.optimizer = torch.optim.SGD


class GLVQ21(GLVQ):
    """Generalized Learning Vector Quantization 2.1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq21_loss
        self.optimizer = torch.optim.SGD


class ImageGLVQ(PrototypeImageModel, GLVQ):
    """GLVQ for training on image data.

    GLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """


class ImageGMLVQ(PrototypeImageModel, GMLVQ):
    """GMLVQ for training on image data.

    GMLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """
