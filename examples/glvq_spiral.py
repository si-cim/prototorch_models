"""GLVQ example using the spiral dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch


class StopOnNaN(pl.Callback):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def on_epoch_end(self, trainer, pl_module, logs={}):
        if torch.isnan(self.param).any():
            raise ValueError("NaN encountered. Stopping.")


if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Spiral(n_samples=600, noise=0.6)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)

    # Hyperparameters
    hparams = dict(
        nclasses=2,
        prototypes_per_class=20,
        prototype_initializer=pt.components.SSI(train_ds, noise=1e-7),
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(hparams)

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=True, block=True)
    snan = StopOnNaN(model.proto_layer.components)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[vis, snan],
    )

    # Training loop
    trainer.fit(model, train_loader)
