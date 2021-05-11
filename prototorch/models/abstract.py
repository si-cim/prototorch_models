import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR


class AbstractPrototypeModel(pl.LightningModule):
    @property
    def prototypes(self):
        return self.proto_layer.components.detach().cpu()

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
