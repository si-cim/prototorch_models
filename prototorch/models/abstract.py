"""Abstract classes to be inherited by prototorch models."""

import pytorch_lightning as pl
import torch
import torchmetrics

from ..core.competitions import WTAC
from ..core.components import Components, LabeledComponents
from ..core.distances import euclidean_distance
from ..core.initializers import LabelsInitializer
from ..core.pooling import stratified_min_pooling
from ..nn.wrappers import LambdaLayer


class ProtoTorchBolt(pl.LightningModule):
    """All ProtoTorch models are ProtoTorch Bolts."""
    def __init__(self, hparams, **kwargs):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters(hparams)

        # Default hparams
        self.hparams.setdefault("lr", 0.01)

        # Default config
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.lr_scheduler_kwargs = kwargs.get("lr_scheduler_kwargs", dict())

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer,
                                          **self.lr_scheduler_kwargs)
            sch = {
                "scheduler": scheduler,
                "interval": "step",
            }  # called after each training step
            return [optimizer], [sch]
        else:
            return optimizer

    def reconfigure_optimizers(self):
        self.trainer.accelerator.setup_optimizers(self.trainer)

    def __repr__(self):
        surep = super().__repr__()
        indented = "".join([f"\t{line}\n" for line in surep.splitlines()])
        wrapped = f"ProtoTorch Bolt(\n{indented})"
        return wrapped


class PrototypeModel(ProtoTorchBolt):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        distance_fn = kwargs.get("distance_fn", euclidean_distance)
        self.distance_layer = LambdaLayer(distance_fn)

    @property
    def num_prototypes(self):
        return len(self.proto_layer.components)

    @property
    def prototypes(self):
        return self.proto_layer.components.detach().cpu()

    @property
    def components(self):
        """Only an alias for the prototypes."""
        return self.prototypes

    def add_prototypes(self, *args, **kwargs):
        self.proto_layer.add_components(*args, **kwargs)
        self.reconfigure_optimizers()

    def remove_prototypes(self, indices):
        self.proto_layer.remove_components(indices)
        self.reconfigure_optimizers()


class UnsupervisedPrototypeModel(PrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Layers
        prototypes_initializer = kwargs.get("prototypes_initializer", None)
        if prototypes_initializer is not None:
            self.proto_layer = Components(
                self.hparams.num_prototypes,
                initializer=prototypes_initializer,
            )

    def compute_distances(self, x):
        protos = self.proto_layer().type_as(x)
        distances = self.distance_layer(x, protos)
        return distances

    def forward(self, x):
        distances = self.compute_distances(x)
        return distances


class SupervisedPrototypeModel(PrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Layers
        prototypes_initializer = kwargs.get("prototypes_initializer", None)
        labels_initializer = kwargs.get("labels_initializer",
                                        LabelsInitializer())
        if prototypes_initializer is not None:
            self.proto_layer = LabeledComponents(
                distribution=self.hparams.distribution,
                components_initializer=prototypes_initializer,
                labels_initializer=labels_initializer,
            )
        self.competition_layer = WTAC()

    @property
    def prototype_labels(self):
        return self.proto_layer.labels.detach().cpu()

    @property
    def num_classes(self):
        return self.proto_layer.num_classes

    def compute_distances(self, x):
        protos, _ = self.proto_layer()
        distances = self.distance_layer(x, protos)
        return distances

    def forward(self, x):
        distances = self.compute_distances(x)
        _, plabels = self.proto_layer()
        winning = stratified_min_pooling(distances, plabels)
        y_pred = torch.nn.functional.softmin(winning)
        return y_pred

    def predict_from_distances(self, distances):
        with torch.no_grad():
            _, plabels = self.proto_layer()
            y_pred = self.competition_layer(distances, plabels)
        return y_pred

    def predict(self, x):
        with torch.no_grad():
            distances = self.compute_distances(x)
        y_pred = self.predict_from_distances(distances)
        return y_pred

    def log_acc(self, distances, targets, tag):
        preds = self.predict_from_distances(distances)
        accuracy = torchmetrics.functional.accuracy(preds.int(), targets.int())
        # `.int()` because FloatTensors are assumed to be class probabilities

        self.log(tag,
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def test_step(self, batch, batch_idx):
        x, targets = batch

        preds = self.predict(x)
        accuracy = torchmetrics.functional.accuracy(preds.int(), targets.int())

        self.log("test_acc", accuracy)


class ProtoTorchMixin(object):
    """All mixins are ProtoTorchMixins."""
    pass


class NonGradientMixin(ProtoTorchMixin):
    """Mixin for custom non-gradient optimization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        raise NotImplementedError


class ImagePrototypesMixin(ProtoTorchMixin):
    """Mixin for models with image prototypes."""
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Constrain the components to the range [0, 1] by clamping after updates."""
        self.proto_layer.components.data.clamp_(0.0, 1.0)

    def get_prototype_grid(self, num_columns=2, return_channels_last=True):
        from torchvision.utils import make_grid
        grid = make_grid(self.components, nrow=num_columns)
        if return_channels_last:
            grid = grid.permute((1, 2, 0))
        return grid.cpu()
