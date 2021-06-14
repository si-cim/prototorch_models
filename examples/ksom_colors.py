"""Kohonen Self Organizing Map."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from prototorch.utils.colors import hex_to_rgb


class Vis2DColorSOM(pl.Callback):
    def __init__(self, data, title="ColorSOMe", pause_time=0.1):
        super().__init__()
        self.title = title
        self.fig = plt.figure(self.title)
        self.data = data
        self.pause_time = pause_time

    def on_epoch_end(self, trainer, pl_module):
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        h, w = pl_module._grid.shape[:2]
        protos = pl_module.prototypes.view(h, w, 3)
        ax.imshow(protos)
        ax.axis("off")

        # Overlay color names
        d = pl_module.compute_distances(self.data)
        wp = pl_module.predict_from_distances(d)
        for i, iloc in enumerate(wp):
            plt.text(iloc[1],
                     iloc[0],
                     cnames[i],
                     ha="center",
                     va="center",
                     bbox=dict(facecolor="white", alpha=0.5, lw=0))

        if trainer.current_epoch != trainer.max_epochs - 1:
            plt.pause(self.pause_time)
        else:
            plt.show(block=True)


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=42)

    # Prepare the data
    hex_colors = [
        "#000000", "#0000ff", "#00007f", "#1f86ff", "#5466aa", "#997fff",
        "#00ff00", "#ff0000", "#00ffff", "#ff00ff", "#ffff00", "#ffffff",
        "#545454", "#7f7f7f", "#a8a8a8", "#808000", "#800080", "#ffa500"
    ]
    cnames = [
        "black", "blue", "darkblue", "skyblue", "greyblue", "lilac", "green",
        "red", "cyan", "magenta", "yellow", "white", "darkgrey", "mediumgrey",
        "lightgrey", "olive", "purple", "orange"
    ]
    colors = list(hex_to_rgb(hex_colors))
    data = torch.Tensor(colors) / 255.0
    train_ds = torch.utils.data.TensorDataset(data)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8)

    # Hyperparameters
    hparams = dict(
        shape=(18, 32),
        alpha=1.0,
        sigma=16,
        lr=0.1,
    )

    # Initialize the model
    model = pt.models.KohonenSOM(
        hparams,
        prototypes_initializer=pt.initializers.RNCI(3),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 3)

    # Model summary
    print(model)

    # Callbacks
    vis = Vis2DColorSOM(data=data)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=500,
        callbacks=[vis],
        weights_summary="full",
    )

    # Training loop
    trainer.fit(model, train_loader)
