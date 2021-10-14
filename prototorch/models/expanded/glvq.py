import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.core.competitions import WTAC, wtac
from prototorch.core.components import Components, LabeledComponents
from prototorch.core.distances import (
    euclidean_distance,
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
)
from prototorch.core.initializers import EyeTransformInitializer, LabelsInitializer
from prototorch.core.losses import GLVQLoss, lvq1_loss, lvq21_loss
from prototorch.core.pooling import stratified_min_pooling
from prototorch.core.transforms import LinearTransform
from prototorch.nn.wrappers import LambdaLayer, LossLayer
from torch.nn.parameter import Parameter


class GLVQ(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters(hparams)

        # Default hparams
        # TODO: Manage by an HPARAMS Object
        self.hparams.setdefault("lr", 0.01)
        self.hparams.setdefault("margin", 0.0)
        self.hparams.setdefault("transfer_fn", "identity")
        self.hparams.setdefault("transfer_beta", 10.0)

        # Default config
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.lr_scheduler_kwargs = kwargs.get("lr_scheduler_kwargs", dict())
        distance_fn = kwargs.get("distance_fn", euclidean_distance)
        prototypes_initializer = kwargs.get("prototypes_initializer", None)
        labels_initializer = kwargs.get("labels_initializer",
                                        LabelsInitializer())

        if prototypes_initializer is not None:
            self.proto_layer = LabeledComponents(
                distribution=self.hparams.distribution,
                components_initializer=prototypes_initializer,
                labels_initializer=labels_initializer,
            )

        self.distance_layer = LambdaLayer(distance_fn)
        self.competition_layer = WTAC()

        self.loss = GLVQLoss(
            margin=self.hparams.margin,
            transfer_fn=self.hparams.transfer_fn,
            beta=self.hparams.transfer_beta,
        )

    def log_acc(self, distances, targets, tag):
        preds = self.predict_from_distances(distances)
        accuracy = torchmetrics.functional.accuracy(preds.int(), targets.int())
        self.log(tag,
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

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
        out, val_loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", val_loss)
        self.log_acc(out, batch[-1], tag="val_acc")
        return val_loss

    def test_step(self, batch, batch_idx):
        out, test_loss = self.shared_step(batch, batch_idx)
        self.log_acc(out, batch[-1], tag="test_acc")
        return test_loss

    def test_epoch_end(self, outputs):
        test_loss = 0.0
        for batch_loss in outputs:
            test_loss += batch_loss.item()
        self.log("test_loss", test_loss)

    # API
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

    @property
    def prototype_labels(self):
        return self.proto_layer.labels.detach().cpu()

    @property
    def num_classes(self):
        return self.proto_layer.num_classes

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

    # Python overwrites
    def __repr__(self):
        surep = super().__repr__()
        indented = "".join([f"\t{line}\n" for line in surep.splitlines()])
        wrapped = f"ProtoTorch Bolt(\n{indented})"
        return wrapped
