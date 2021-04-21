"""GLVQ example using the MNIST dataset.

TODO
- Add model serialization/deserialization
- Add evaluation metrics

This script also shows how to use Tensorboard for visualizing the prototypes.
"""

import argparse

import pytorch_lightning as pl
import torchvision
from matplotlib import pyplot as plt
from prototorch.functions.initializers import stratified_mean
from prototorch.models.glvq import ImageGLVQ
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class VisualizationCallback(pl.Callback):
    def __init__(self, to_shape=(-1, 1, 28, 28), nrow=2):
        super().__init__()
        self.to_shape = to_shape
        self.nrow = nrow

    def on_epoch_end(self, trainer, pl_module):
        protos = pl_module.proto_layer.prototypes.detach().cpu()
        protos_img = protos.reshape(self.to_shape)
        grid = torchvision.utils.make_grid(protos_img, nrow=self.nrow)
        # grid = grid.permute((1, 2, 0))
        tb = pl_module.logger.experiment
        tb.add_image(tag="MNIST Prototypes",
                     img_tensor=grid,
                     global_step=trainer.current_epoch,
                     dataformats="CHW")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Epochs to train.")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="Batch size.")
    parser.add_argument("--gpus",
                        type=int,
                        default=0,
                        help="Number of GPUs to use.")
    parser.add_argument("--ppc",
                        type=int,
                        default=1,
                        help="Prototypes-Per-Class.")
    args = parser.parse_args()

    # Dataset
    mnist_train = MNIST(
        "./datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
    )
    mnist_test = MNIST(
        "./datasets",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
    )

    # Dataloaders
    train_loader = DataLoader(mnist_train, batch_size=1024)
    test_loader = DataLoader(mnist_test, batch_size=1024)

    # Grab the full dataset to warm-start prototypes
    x, y = next(iter(DataLoader(mnist_train, batch_size=len(mnist_train))))
    x = x.view(len(mnist_train), -1)

    # Initialize the model
    model = ImageGLVQ(input_dim=28 * 28,
                      nclasses=10,
                      prototypes_per_class=args.ppc,
                      prototype_initializer="stratified_mean",
                      data=[x, y])
    # Model summary
    print(model)

    # Callbacks
    vis = VisualizationCallback(to_shape=(-1, 1, 28, 28), nrow=args.ppc)

    # Setup trainer
    trainer = pl.Trainer(
        gpus=args.gpus,  # change to use GPUs for training
        max_epochs=args.epochs,
        callbacks=[vis],
        # accelerator="ddp_cpu",  # DEBUG-ONLY
        # num_processes=2,  # DEBUG-ONLY
    )

    # Training loop
    trainer.fit(model, train_loader, test_loader)
