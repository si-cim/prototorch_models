"""Warm-starting GLVQ with prototypes from Growing Neural Gas."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import (
    GLVQ,
    KNN,
    GrowingNeuralGas,
    PruneLoserPrototypes,
    VisGLVQ2D,
)
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)

if __name__ == "__main__":

    # Reproducibility
    seed_everything(seed=4)
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Prepare the data
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_loader = DataLoader(train_ds, batch_size=64, num_workers=0)

    # Initialize the gng
    gng = GrowingNeuralGas(
        hparams=dict(num_prototypes=5, insert_freq=2, lr=0.1),
        prototypes_initializer=pt.initializers.ZCI(2),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Callbacks
    es = EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=20,
        mode="min",
        verbose=False,
        check_on_train_epoch_end=True,
    )

    # Setup trainer for GNG
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=50 if args.fast_dev_run else
        1000,  # 10 epochs fast dev run reproducible DIV error.
        callbacks=[
            es,
        ],
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(gng, train_loader)

    # Hyperparameters
    hparams = dict(
        distribution=[],
        lr=0.01,
    )

    # Warm-start prototypes
    knn = KNN(dict(k=1), data=train_ds)
    prototypes = gng.prototypes
    plabels = knn.predict(prototypes)

    # Initialize the model
    model = GLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.LCI(prototypes),
        labels_initializer=pt.initializers.LLI(plabels),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)
    pruning = PruneLoserPrototypes(
        threshold=0.02,
        idle_epochs=2,
        prune_quota_per_epoch=5,
        frequency=1,
        verbose=True,
    )
    es = EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=10,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
            pruning,
            es,
        ],
        max_epochs=1000,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
