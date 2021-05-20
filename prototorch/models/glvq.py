import torch
import torchmetrics
from prototorch.components import LabeledComponents
from prototorch.functions.activations import get_activation
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import (euclidean_distance, omega_distance,
                                            sed)
from prototorch.functions.helper import get_flat
from prototorch.functions.losses import (_get_dp_dm, _get_matcher, glvq_loss,
                                         lvq1_loss, lvq21_loss)

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
        self.hparams.setdefault("transfer_fn", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)
        self.hparams.setdefault("lr", 0.01)

        self.proto_layer = LabeledComponents(
            distribution=self.hparams.distribution,
            initializer=prototype_initializer)

        self.transfer_fn = get_activation(self.hparams.transfer_fn)
        self.acc_metric = torchmetrics.Accuracy()

        self.loss = glvq_loss

    @property
    def prototype_labels(self):
        return self.proto_layer.component_labels.detach().cpu()

    def forward(self, x):
        protos, _ = self.proto_layer()
        distances = self.distance_fn(x, protos)
        return distances

    def log_acc(self, distances, targets, tag):
        plabels = self.proto_layer.component_labels
        # Compute training accuracy
        with torch.no_grad():
            preds = wtac(distances, plabels)

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
        out = self(x)
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
        total_loss = 0
        for batch_loss in outputs:
            total_loss += batch_loss.item()
        self.log('test_loss', total_loss)

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     pass

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            d = self(x)
            plabels = self.proto_layer.component_labels
            y_pred = wtac(d, plabels)
        return y_pred

    def __repr__(self):
        super_repr = super().__repr__()
        return f"{super_repr}"


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


class NonGradientGLVQ(GLVQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        raise NotImplementedError


class LVQ1(NonGradientGLVQ):
    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos = self.proto_layer.components
        plabels = self.proto_layer.component_labels

        x, y = train_batch
        dis = self(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            d = self(xi.view(1, -1))
            preds = wtac(d, plabels)
            w = d.argmin(1)
            if yi == preds:
                shift = xi - protos[w]
            else:
                shift = protos[w] - xi
            updated_protos = protos + 0.0
            updated_protos[w] = protos[w] + (self.hparams.lr * shift)
            self.proto_layer.load_state_dict({"_components": updated_protos},
                                             strict=False)

        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None


class LVQ21(NonGradientGLVQ):
    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos = self.proto_layer.components
        plabels = self.proto_layer.component_labels

        x, y = train_batch
        dis = self(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            xi = xi.view(1, -1)
            yi = yi.view(1, )
            d = self(xi)
            preds = wtac(d, plabels)
            (dp, wp), (dn, wn) = _get_dp_dm(d, yi, plabels, with_indices=True)
            shiftp = xi - protos[wp]
            shiftn = protos[wn] - xi
            updated_protos = protos + 0.0
            updated_protos[wp] = protos[wp] + (self.hparams.lr * shiftp)
            updated_protos[wn] = protos[wn] + (self.hparams.lr * shiftn)
            self.proto_layer.load_state_dict({"_components": updated_protos},
                                             strict=False)

        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None


class MedianLVQ(NonGradientGLVQ):
    ...


class GLVQ1(GLVQ):
    """Learning Vector Quantization 1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = lvq1_loss
        self.optimizer = torch.optim.SGD


class GLVQ21(GLVQ):
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
