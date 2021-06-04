import pytorch_lightning as pl


class AbstractPrototypeModel(pl.LightningModule):
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


class PrototypeImageModel(pl.LightningModule):
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.proto_layer.components.data.clamp_(0.0, 1.0)

    def get_prototype_grid(self, num_columns=2, return_channels_last=True):
        from torchvision.utils import make_grid
        grid = make_grid(self.components, nrow=num_columns)
        if return_channels_last:
            grid = grid.permute((1, 2, 0))
        return grid.cpu()
