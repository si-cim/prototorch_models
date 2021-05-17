"""LVQMLN example using all four dimensions of the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

from siamese_glvq_iris import Backbone

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Iris()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)

    # Hyperparameters
    hparams = dict(
        distribution=[1, 2, 2],
        proto_lr=0.001,
        bb_lr=0.001,
    )

    # Initialize the backbone
    backbone = Backbone()

    # Initialize the model
    model = pt.models.LVQMLN(
        hparams,
        prototype_initializer=pt.components.SSI(train_ds, transform=backbone),
        backbone=backbone,
    )

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisSiameseGLVQ2D(
        data=train_ds,
        map_protos=False,
        border=0.1,
        resolution=500,
        axis_off=True,
    )

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis], gpus=0)

    # Training loop
    trainer.fit(model, train_loader)
