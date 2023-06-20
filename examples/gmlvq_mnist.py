"""GMLVQ example using the MNIST dataset."""

import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import (
    ImageGMLVQ,
    PruneLoserPrototypes,
    VisImgComp,
)
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Dataset
    train_ds = MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    test_ds = MNIST(
        "~/datasets",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=4, batch_size=256)
    test_loader = DataLoader(test_ds, num_workers=4, batch_size=256)

    # Hyperparameters
    num_classes = 10
    prototypes_per_class = 10
    hparams = dict(
        input_dim=28 * 28,
        latent_dim=28 * 28,
        distribution=(num_classes, prototypes_per_class),
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = ImageGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
    )

    # Callbacks
    vis = VisImgComp(
        data=train_ds,
        num_columns=10,
        show=False,
        tensorboard=True,
        random_data=100,
        add_embedding=True,
        embedding_data=200,
        flatten_data=False,
    )
    pruning = PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=1,
        prune_quota_per_epoch=10,
        frequency=1,
        verbose=True,
    )
    es = EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=15,
        mode="min",
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
