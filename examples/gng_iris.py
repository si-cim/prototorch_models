"""Growing Neural Gas example using the Iris dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Prepare the data
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    # Hyperparameters
    hparams = dict(
        num_prototypes=5,
        input_dim=2,
        lr=0.1,
    )

    # Initialize the model
    model = pt.models.GrowingNeuralGas(
        hparams,
        prototypes_initializer=pt.initializers.ZCI(2),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisNG2D(data=train_loader)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        callbacks=[vis],
        weights_summary="full",
    )

    # Training loop
    trainer.fit(model, train_loader)
