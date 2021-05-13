"""GLVQ example using the spiral dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.models.callbacks import StopOnNaN

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Spiral(n_samples=600, noise=0.6)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)

    # Hyperparameters
    nclasses = 2
    prototypes_per_class = 20
    hparams = dict(
        distribution=(nclasses, prototypes_per_class),
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GLVQ(hparams,
                           prototype_initializer=pt.components.SSI(train_ds,
                                                                   noise=1e-1))

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=True)
    snan = StopOnNaN(model.proto_layer.components)

    # Setup trainer
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=200,
        callbacks=[vis, snan],
    )

    # Training loop
    trainer.fit(model, train_loader)
