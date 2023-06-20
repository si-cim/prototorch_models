"""GLVQ example using the Iris dataset."""

import argparse
import logging
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import GLVQ, VisGLVQ2D
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64, num_workers=4)

    # Hyperparameters
    hparams = dict(
        distribution={
            "num_classes": 3,
            "per_class": 4
        },
        lr=0.01,
    )

    # Initialize the model
    model = GLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Callbacks
    vis = VisGLVQ2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
        ],
        max_epochs=100,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)

    # Manual save
    trainer.save_checkpoint("./glvq_iris.ckpt")

    # Load saved model
    new_model = GLVQ.load_from_checkpoint(
        checkpoint_path="./glvq_iris.ckpt",
        strict=False,
    )
    logging.info(new_model)
