import pytorch_lightning as pl
import torch
import torchmetrics

from prototorch.functions.competitions import wtac
from prototorch.functions.distances import euclidean_distance
from prototorch.functions.losses import glvq_loss
from prototorch.modules.prototypes import Prototypes1D


class GLVQ(pl.LightningModule):
    """Generalized Learning Vector Quantization."""
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.proto_layer = Prototypes1D(
            input_dim=self.hparams.input_dim,
            nclasses=self.hparams.nclasses,
            prototypes_per_class=self.hparams.prototypes_per_class,
            prototype_initializer=self.hparams.prototype_initializer,
            **kwargs)
        self.train_acc = torchmetrics.Accuracy()

    @property
    def prototypes(self):
        return self.proto_layer.prototypes.detach().numpy()

    @property
    def prototype_labels(self):
        return self.proto_layer.prototype_labels.detach().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        protos = self.proto_layer.prototypes
        dis = euclidean_distance(x, protos)
        return dis

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        dis = self(x)
        plabels = self.proto_layer.prototype_labels
        mu = glvq_loss(dis, y, prototype_labels=plabels)
        loss = mu.sum(dim=0)
        self.log("train_loss", loss)
        with torch.no_grad():
            preds = wtac(dis, plabels)
        # self.train_acc.update(preds.int(), y.int())
        self.train_acc(
            preds.int(),
            y.int())  # FloatTensors are assumed to be class probabilities
        self.log(
            "acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    # def training_epoch_end(self, outs):
    #     # Calling `self.train_acc.compute()` is
    #     # automatically done by setting `on_epoch=True` when logging in `self.training_step(...)`
    #     self.log("train_acc_epoch", self.train_acc.compute())

    def predict(self, x):
        # model.eval()  # ?!
        with torch.no_grad():
            d = self(x)
            plabels = self.proto_layer.prototype_labels
            y_pred = wtac(d, plabels)
        return y_pred.numpy()


class ImageGLVQ(GLVQ):
    """GLVQ for training on image data.

    GLVQ model that constrains the prototypes to the range [0, 1] by
    clamping after updates.
    """
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.proto_layer.prototypes.data.clamp_(0.0, 1.0)


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
                 **kwargs):
        super().__init__(hparams, **kwargs)
        self.backbone = backbone_module(**backbone_params)
        self.backbone_dependent = backbone_module(
            **backbone_params).requires_grad_(False)

    def sync_backbones(self):
        master_state = self.backbone.state_dict()
        self.backbone_dependent.load_state_dict(master_state, strict=True)

    def forward(self, x):
        self.sync_backbones()
        protos = self.proto_layer.prototypes

        latent_x = self.backbone(x)
        latent_protos = self.backbone_dependent(protos)

        dis = euclidean_distance(latent_x, latent_protos)
        return dis

    def predict_latent(self, x):
        # model.eval()  # ?!
        with torch.no_grad():
            protos = self.proto_layer.prototypes
            latent_protos = self.backbone_dependent(protos)
            d = euclidean_distance(x, latent_protos)
            plabels = self.proto_layer.prototype_labels
            y_pred = wtac(d, plabels)
        return y_pred.numpy()
