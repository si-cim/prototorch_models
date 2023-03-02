"""Models based on the GLVQ framework."""

import torch
from prototorch.core.competitions import wtac
from prototorch.core.distances import (
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
)
from prototorch.core.initializers import EyeLinearTransformInitializer
from prototorch.core.losses import (
    GLVQLoss,
    lvq1_loss,
    lvq21_loss,
)
from prototorch.core.transforms import LinearTransform
from prototorch.nn.wrappers import LambdaLayer, LossLayer
from torch.nn.parameter import Parameter

from .abstract import ImagePrototypesMixin, SupervisedPrototypeModel
from .extras import ltangent_distance, orthogonalization


class GLVQ(SupervisedPrototypeModel):
    """Generalized Learning Vector Quantization."""

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Default hparams
        self.hparams.setdefault("margin", 0.0)
        self.hparams.setdefault("transfer_fn", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)

        # Loss
        self.loss = GLVQLoss(
            margin=self.hparams["margin"],
            transfer_fn=self.hparams["transfer_fn"],
            beta=self.hparams["transfer_beta"],
        )

    # def on_save_checkpoint(self, checkpoint):
    #     if "prototype_win_ratios" in checkpoint["state_dict"]:
    #         del checkpoint["state_dict"]["prototype_win_ratios"]

    def initialize_prototype_win_ratios(self):
        self.register_buffer(
            "prototype_win_ratios",
            torch.zeros(self.num_prototypes, device=self.device))

    def on_train_epoch_start(self):
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
        _, plabels = self.proto_layer()
        loss = self.loss(out, y, plabels)
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
                                   lr=self.hparams["proto_lr"])
        # Only add a backbone optimizer if backbone has trainable parameters
        bb_params = list(self.backbone.parameters())
        if (bb_params):
            bb_opt = self.optimizer(bb_params, lr=self.hparams["bb_lr"])
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
        x, protos = (arr.view(arr.size(0), -1) for arr in (x, protos))
        latent_x = self.backbone(x)

        bb_grad = any([el.requires_grad for el in self.backbone.parameters()])

        self.backbone.requires_grad_(bb_grad and self.both_path_gradients)
        latent_protos = self.backbone(protos)
        self.backbone.requires_grad_(bb_grad)

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
    _relevances: torch.Tensor

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Additional parameters
        relevances = torch.ones(self.hparams["input_dim"], device=self.device)
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
        omega_initializer = kwargs.get("omega_initializer",
                                       EyeLinearTransformInitializer())
        self.backbone = LinearTransform(
            self.hparams["input_dim"],
            self.hparams["latent_dim"],
            initializer=omega_initializer,
        )

    @property
    def omega_matrix(self):
        return self.backbone.weights

    @property
    def lambda_matrix(self):
        omega = self.backbone.weights  # (input_dim, latent_dim)
        lam = omega @ omega.T
        return lam.detach().cpu()


class GMLVQ(GLVQ):
    """Generalized Matrix Learning Vector Quantization.

    Implemented as a regular GLVQ network that simply uses a different distance
    function. This makes it easier to implement a localized variant.

    """

    # Parameters
    _omega: torch.Tensor

    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Additional parameters
        omega_initializer = kwargs.get("omega_initializer",
                                       EyeLinearTransformInitializer())
        omega = omega_initializer.generate(self.hparams["input_dim"],
                                           self.hparams["latent_dim"])
        self.register_parameter("_omega", Parameter(omega))
        self.backbone = LambdaLayer(lambda x: x @ self._omega,
                                    name="omega matrix")

    @property
    def omega_matrix(self):
        return self._omega.detach().cpu()

    @property
    def lambda_matrix(self):
        omega = self._omega.detach()  # (input_dim, latent_dim)
        lam = omega @ omega.T
        return lam.detach().cpu()

    def compute_distances(self, x):
        protos, _ = self.proto_layer()
        distances = self.distance_layer(x, protos, self._omega)
        return distances

    def extra_repr(self):
        return f"(omega): (shape: {tuple(self._omega.shape)})"


class LGMLVQ(GMLVQ):
    """Localized and Generalized Matrix Learning Vector Quantization."""

    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", lomega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Re-register `_omega` to override the one from the super class.
        omega = torch.randn(
            self.num_prototypes,
            self.hparams["input_dim"],
            self.hparams["latent_dim"],
            device=self.device,
        )
        self.register_parameter("_omega", Parameter(omega))


class GTLVQ(LGMLVQ):
    """Localized and Generalized Tangent Learning Vector Quantization."""

    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", ltangent_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        omega_initializer = kwargs.get("omega_initializer")

        if omega_initializer is not None:
            subspace = omega_initializer.generate(
                self.hparams["input_dim"],
                self.hparams["latent_dim"],
            )
            omega = torch.repeat_interleave(
                subspace.unsqueeze(0),
                self.num_prototypes,
                dim=0,
            )
        else:
            omega = torch.rand(
                self.num_prototypes,
                self.hparams["input_dim"],
                self.hparams["latent_dim"],
                device=self.device,
            )

        # Re-register `_omega` to override the one from the super class.
        self.register_parameter("_omega", Parameter(omega))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            self._omega.copy_(orthogonalization(self._omega))


class SiameseGTLVQ(SiameseGLVQ, GTLVQ):
    """Generalized Tangent Learning Vector Quantization.

    Implemented as a Siamese network with a linear transformation backbone.

    """


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


class ImageGTLVQ(ImagePrototypesMixin, GTLVQ):
    """GTLVQ for training on image data.

    GTLVQ model that constrains the prototypes to the range [0, 1] by clamping
    after updates.

    """

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Constrain the components to the range [0, 1] by clamping after updates."""
        self.proto_layer.components.data.clamp_(0.0, 1.0)
        with torch.no_grad():
            self._omega.copy_(orthogonalization(self._omega))
