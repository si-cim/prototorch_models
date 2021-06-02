"""Dynamically prune prototypes in GLVQ-type models."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class PrototypePruning(Callback):
    def __init__(self, threshold=0.01, prune_after=10, verbose=False):
        self.threshold = threshold
        self.prune_after = prune_after
        self.verbose = verbose

    def on_epoch_start(self, trainer, pl_module):
        pl_module.initialize_prototype_win_ratios()

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) > self.prune_after:
            ratios = pl_module.prototype_win_ratios.mean(dim=0)
            to_prune = torch.arange(len(ratios))[ratios < self.threshold]
            if len(to_prune) > 0:
                if self.verbose:
                    print(f"\nPrototype win ratios: {ratios}")
                    print(f"Pruning prototypes at indices: {to_prune}")
                cur_num_protos = pl_module.num_prototypes
                pl_module.remove_prototypes(indices=to_prune)
                new_num_protos = pl_module.num_prototypes
                if self.verbose:
                    print(f"`num_prototypes` reduced from {cur_num_protos} "
                          f"to {new_num_protos}.")


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 4
    num_features = 2
    num_clusters = 1
    train_ds = pt.datasets.Random(num_samples=500,
                                  num_classes=num_classes,
                                  num_features=num_features,
                                  num_clusters=num_clusters,
                                  separation=3.0,
                                  seed=42)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256)

    # Hyperparameters
    prototypes_per_class = num_clusters * 5
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        lr=0.3,
    )

    # Initialize the model
    model = pt.models.CELVQ(
        hparams,
        prototype_initializer=pt.components.Ones(2, scale=3),
    )

    # Summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds)
    pruning = PrototypePruning(
        threshold=0.01,  # prune prototype if it wins less than 1%
        prune_after=50,
        verbose=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        callbacks=[
            vis,
            pruning,
        ],
        terminate_on_nan=True,
        weights_summary=None,
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
