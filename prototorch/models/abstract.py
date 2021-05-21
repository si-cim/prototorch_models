import pytorch_lightning as pl
import torch
from prototorch.functions.competitions import wtac
from torch.optim.lr_scheduler import ExponentialLR


class AbstractPrototypeModel(pl.LightningModule):
    @property
    def prototypes(self):
        return self.proto_layer.components.detach().cpu()

    @property
    def components(self):
        """Only an alias for the prototypes."""
        return self.prototypes

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer,
                                  gamma=0.99,
                                  last_epoch=-1,
                                  verbose=False)
        sch = {
            "scheduler": scheduler,
            "interval": "step",
        }  # called after each training step
        return [optimizer], [sch]


class PrototypeImageModel(pl.LightningModule):
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.proto_layer.components.data.clamp_(0.0, 1.0)

    def get_prototype_grid(self, nrow=2, return_channels_last=True):
        from torchvision.utils import make_grid
        grid = make_grid(self.components, nrow=nrow)
        if return_channels_last:
            grid = grid.permute((1, 2, 0))
        return grid.cpu()


class SiamesePrototypeModel(pl.LightningModule):
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
