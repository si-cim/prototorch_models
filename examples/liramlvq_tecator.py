"""Limited Rank Matrix LVQ example using the Tecator dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Tecator(root="~/datasets/", train=True)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=32)

    # Hyperparameters
    hparams = dict(
        nclasses=2,
        prototypes_per_class=2,
        input_dim=100,
        latent_dim=2,
        prototype_initializer=pt.components.SMI(train_ds),
        lr=0.001,
    )

    # Initialize the model
    model = pt.models.GMLVQ(hparams)

    # Callbacks
    vis = pt.models.VisSiameseGLVQ2D(train_ds, border=0.1)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=200, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
