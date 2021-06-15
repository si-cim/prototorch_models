"""Models based on the GLVQ framework."""

import torch
from prototorch.functions.activations import get_activation
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import (
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
)
from prototorch.functions.helper import get_flat
from prototorch.functions.losses import glvq_loss, lvq1_loss, lvq21_loss
from prototorch.modules import LambdaLayer, LossLayer
from torch.nn.parameter import Parameter

from .abstract import ImagePrototypesMixin, SupervisedPrototypeModel


class GLVQ(SupervisedPrototypeModel):
    """Generalized Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Default hparams
        self.hparams.setdefault("transfer_fn", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)

        # Layers
        transfer_fn = get_activation(self.hparams.transfer_fn)
        self.transfer_layer = LambdaLayer(transfer_fn)

        # Loss
        self.loss = LossLayer(glvq_loss)

        # Prototype metrics
        self.initialize_prototype_win_ratios()

    def initialize_prototype_win_ratios(self):
        self.register_buffer(
            "prototype_win_ratios",
            torch.zeros(self.num_prototypes, device=self.device))

    def on_epoch_start(self):
        self.initialize_prototype_win_ratios()

    def log_prototype_win_ratios(self, distances):
        batch_size = len(distances)
        prototype_wc = torch.zeros(self.num_prototypes,
                                   dtype=torch.long,
                                   device=self.device)
        wi, wc = torch.unique(distances.min(dim=-1).indices,
                              sorted=True,
                              return_counts=True)
        prototype_wc[wi] = wc
        prototype_wr = prototype_wc / batch_size
        self.prototype_win_ratios = torch.vstack([
            self.prototype_win_ratios,
            prototype_wr,
        ])

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.component_labels
        mu = self.loss(out, y, prototype_labels=plabels)
        batch_loss = self.transfer_layer(mu, beta=self.hparams.transfer_beta)
        loss = batch_loss.sum(dim=0)
        return out, loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        out, train_loss = self.shared_step(batch, batch_idx, optimizer_idx)
        self.log_prototype_win_ratios(out)
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

    # TODO
    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     pass


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
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        self.backbone = backbone
        self.both_path_gradients = both_path_gradients

    def configure_optimizers(self):
        proto_opt = self.optimizer(self.proto_layer.parameters(),
                                   lr=self.hparams.proto_lr)
        # Only add a backbone optimizer if backbone has trainable parameters
        if (bb_params := list(self.backbone.parameters())):
            bb_opt = self.optimizer(bb_params, lr=self.hparams.bb_lr)
            optimizers = [proto_opt, bb_opt]
        else:
            optimizers = [proto_opt]
        if self.lr_scheduler is not None:
            schedulers = []
            for optimizer in optimizers:
                scheduler = self.lr_scheduler(optimizer,
                                              **self.lr_scheduler_kwargs)
                schedulers.append(scheduler)
            return optimizers, schedulers
        else:
            return optimizers

    def compute_distances(self, x):
        protos, _ = self.proto_layer()
        x, protos = get_flat(x, protos)
        latent_x = self.backbone(x)
        self.backbone.requires_grad_(self.both_path_gradients)
        latent_protos = self.backbone(protos)
        self.backbone.requires_grad_(True)
        distances = self.distance_layer(latent_x, latent_protos)
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
            d = self.distance_layer(x, protos)
            y_pred = wtac(d, plabels)
        return y_pred


class LVQMLN(SiameseGLVQ):
    """Learning Vector Quantization Multi-Layer Network.

    GLVQ model that applies an arbitrary transformation on the inputs, BUT NOT
    on the prototypes before computing the distances between them. This of
    course, means that the prototypes no longer live the input space, but
    rather in the embedding space.

    """
    def compute_distances(self, x):
        latent_protos, _ = self.proto_layer()
        latent_x = self.backbone(x)
        distances = self.distance_layer(latent_x, latent_protos)
        return distances


class GRLVQ(SiameseGLVQ):
    """Generalized Relevance Learning Vector Quantization.

    Implemented as a Siamese network with a linear transformation backbone.

    TODO Make a RelevanceLayer. `bb_lr` is ignored otherwise.

    """
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Additional parameters
        relevances = torch.ones(self.hparams.input_dim, device=self.device)
        self.register_parameter("_relevances", Parameter(relevances))

        # Override the backbone
        self.backbone = LambdaLayer(lambda x: x @ torch.diag(self._relevances),
                                    name="relevance scaling")

    @property
    def relevance_profile(self):
        return self._relevances.detach().cpu()

    def extra_repr(self):
        return f"(relevances): (shape: {tuple(self._relevances.shape)})"


class SiameseGMLVQ(SiameseGLVQ):
    """Generalized Matrix Learning Vector Quantization.

    Implemented as a Siamese network with a linear transformation backbone.

    """
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Override the backbone
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


class GMLVQ(GLVQ):
    """Generalized Matrix Learning Vector Quantization.

    Implemented as a regular GLVQ network that simply uses a different distance
    function. This makes it easier to implement a localized variant.

    """
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Additional parameters
        omega = torch.randn(self.hparams.input_dim,
                            self.hparams.latent_dim,
                            device=self.device)
        self.register_parameter("_omega", Parameter(omega))
        self.backbone = LambdaLayer(lambda x: x @ self._omega, name = "omega matrix")

    @property
    def omega_matrix(self):
        return self._omega.detach().cpu()

    def compute_distances(self, x):
        protos, _ = self.proto_layer()
        distances = self.distance_layer(x, protos, self._omega)
        return distances

    def extra_repr(self):
        return f"(omega): (shape: {tuple(self._omega.shape)})"

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
            print(x, protos)
            d = squared_euclidean_distance(x, protos)
            y_pred = wtac(d, plabels)
        return y_pred



class LGMLVQ(GMLVQ):
    """Localized and Generalized Matrix Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", lomega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Re-register `_omega` to override the one from the super class.
        omega = torch.randn(
            self.num_prototypes,
            self.hparams.input_dim,
            self.hparams.latent_dim,
            device=self.device,
        )
        self.register_parameter("_omega", Parameter(omega))


class GLVQ1(GLVQ):
    """Generalized Learning Vector Quantization 1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = LossLayer(lvq1_loss)
        self.optimizer = torch.optim.SGD


class GLVQ21(GLVQ):
    """Generalized Learning Vector Quantization 2.1."""
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss = LossLayer(lvq21_loss)
        self.optimizer = torch.optim.SGD


class ImageGLVQ(ImagePrototypesMixin, GLVQ):
    """GLVQ for training on image data.

    GLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """


class ImageGMLVQ(ImagePrototypesMixin, GMLVQ):
    """GMLVQ for training on image data.

    GMLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """
