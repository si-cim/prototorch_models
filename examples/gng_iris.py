"""Growing Neural Gas example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
from prototorch.components.initializers import Zeros
from prototorch.datasets import Iris
from prototorch.models.unsupervised import GrowingNeuralGas
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Prepare the data
    train_ds = Iris(dims=[0, 2])
    train_loader = DataLoader(train_ds, batch_size=8)

    # Hyperparameters
    hparams = dict(
        num_prototypes=5,
        lr=0.1,
    )

    # Initialize the model
    model = GrowingNeuralGas(
        hparams,
        prototype_initializer=Zeros(2),
    )

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisNG2D(data=train_loader)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)

    # Model summary
    print(model)
