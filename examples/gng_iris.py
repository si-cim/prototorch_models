import argparse

import prototorch as pt
import pytorch_lightning as pl
from prototorch.components.initializers import SelectionInitializer
from prototorch.datasets import Iris
from prototorch.models.unsupervised import GrowingNeuralGas
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Prepare the data
    train_ds = Iris(dims=[0, 2])
    train_loader = DataLoader(train_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(num_prototypes=2,
                   lr=0.1,
                   prototype_initializer=SelectionInitializer(train_ds.data))

    # Initialize the model
    model = GrowingNeuralGas(hparams)

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
