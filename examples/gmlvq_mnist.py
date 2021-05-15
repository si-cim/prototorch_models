"""GMLVQ example using the MNIST dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == "__main__":
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
    nclasses = 10
    prototypes_per_class = 2
    hparams = dict(
        input_dim=28 * 28,
        latent_dim=28 * 28,
        distribution=(nclasses, prototypes_per_class),
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.ImageGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototype_initializer=pt.components.SMI(train_ds),
    )

    # Callbacks
    vis = pt.models.VisImgComp(data=train_ds,
                               nrow=5,
                               show=True,
                               tensorboard=True,
                               pause_time=0.5)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[vis],
        gpus=0,
        # overfit_batches=1,
        # fast_dev_run=3,
    )

    # Training loop
    trainer.fit(model, train_loader)
