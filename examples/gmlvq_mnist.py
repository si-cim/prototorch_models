"""GMLVQ example using the MNIST dataset."""

import argparse

import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

import prototorch as pt

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
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
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              num_workers=0,
                                              batch_size=256)

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
    model = pt.models.ImageGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototype_initializer=pt.components.SMI(train_ds),
    )

    # Callbacks
    vis = pt.models.VisImgComp(
        data=train_ds,
        num_columns=10,
        show=False,
        tensorboard=True,
        random_data=100,
        add_embedding=True,
        embedding_data=200,
        flatten_data=False,
    )
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=1,
        prune_quota_per_epoch=10,
        frequency=1,
        verbose=True,
    )
    es = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=15,
        mode="min",
        check_on_train_epoch_end=True,
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            pruning,
            # es,
        ],
        terminate_on_nan=True,
        weights_summary=None,
        accelerator="ddp",
    )

    # Training loop
    trainer.fit(model, train_loader)
