"""Limited Rank MLVQ example using the Tecator dataset."""

import pytorch_lightning as pl
from prototorch.components import initializers as cinit
from prototorch.datasets.tecator import Tecator
from torch.utils.data import DataLoader

from prototorch.models.callbacks.visualization import VisSiameseGLVQ2D
from prototorch.models.glvq import GMLVQ

if __name__ == "__main__":
    # Dataset
    train_ds = Tecator(root="./datasets/", train=True)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=32)

    # Grab the full dataset to warm-start prototypes
    x, y = next(iter(DataLoader(train_ds, batch_size=len(train_ds))))

    # Hyperparameters
    hparams = dict(
        nclasses=2,
        prototypes_per_class=2,
        prototype_initializer=cinit.SMI(x, y),
        input_dim=x.shape[1],
        latent_dim=2,
        lr=0.01,
    )

    # Initialize the model
    model = GMLVQ(hparams)

    # Model summary
    print(model)

    # Callbacks
    vis = VisSiameseGLVQ2D(x, y)

    # Namespace hook for the visualization to work
    model.backbone = model.omega_layer

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
