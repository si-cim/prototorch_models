"""Neural Additive Model (NAM) example for binary classification."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.XOR()

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    hparams = dict(lr=0.001)

    # Define the feature extractor
    class FE(torch.nn.Module):
        def __init__(self, hidden_size=10):
            super().__init__()
            self.modules_list = torch.nn.ModuleList([
                torch.nn.Linear(1, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1),
                torch.nn.ReLU(),
            ])

        def forward(self, x):
            for m in self.modules_list:
                x = m(x)
            return x

    # Initialize the model
    model = pt.models.BinaryNAM(
        hparams,
        extractors=torch.nn.ModuleList([FE(20) for _ in range(2)]),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.Vis2D(data=train_ds)
    es = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=50,
        mode="min",
        verbose=False,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            es,
        ],
        terminate_on_nan=True,
        weights_summary="full",
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)

    # Visualize extractor shape functions
    fig, axes = plt.subplots(2)
    for i, ax in enumerate(axes.flat):
        x = torch.linspace(0, 1, 100)  # TODO use min/max from data
        y = model.extractors[i](x.view(100, 1)).squeeze().detach()
        ax.plot(x, y)
        ax.set(title=f"Feature {i + 1}")
    plt.show()
