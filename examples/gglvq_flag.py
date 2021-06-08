"""GLVQ example using the spiral dataset."""

import argparse

import pytorch_lightning as pl
import torch
import numpy as np

import prototorch as pt

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_samples=300
    dimension=2
    num_classes = 1
    x = np.random.uniform(low=0.0,high=1.0,size=(num_samples,dimension))
    y = np.full(num_samples, 0, dtype=int)
    i = np.where(np.sqrt(((x[:,0]-0.5)**4 + (x[:,1]-0.5)**2)) < 0.15)
    y[i] = 1

    print(x.shape, y.shape)

    #train_ds = pt.datasets.Spiral(num_samples=600, noise=0.6)
    train_ds = pt.datasets.NumpyDataset(x, y)


    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)
    """
    # Hyperparameters
    num_classes = 1
    prototypes_per_class = 2
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.OneClassGLVQ(hparams,
                                   #prototype_initializer=pt.components.SMI(train_ds))
                                   prototype_initializer=pt.components.Random(x.shape[1]))
    """
    # Hyperparameters
    num_classes = 1
    prototypes_per_class = 4
    hparams = dict(
        input_dim=2,
        latent_dim=2,
        distribution=(num_classes, prototypes_per_class),
        transfer_function="sigmoid_beta",
        transfer_beta=10.0,
        proto_lr=0.01,
        bb_lr=0.01
    )

    # Initialize the model
    model = pt.models.OneClassGMLVQ(hparams,
                                   prototype_initializer=pt.components.SMI(train_ds))
                                   #prototype_initializer=pt.components.Random(x.shape[1]))

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=False)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[vis],
        terminate_on_nan=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
