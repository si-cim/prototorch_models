import pytorch_lightning as pl
import torch


class AbstractLightningModel(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class AbstractPrototypeModel(AbstractLightningModel):
    @property
    def prototypes(self):
        return self.proto_layer.components.detach().numpy()
