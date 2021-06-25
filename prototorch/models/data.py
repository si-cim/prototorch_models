"""Prototorch Data Modules

This allows to store the used dataset inside a Lightning Module.
Mainly used for PytorchLightningCLI configurations.
"""
from typing import Any, Optional, Type

import prototorch as pt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


# MNIST
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    # Download mnist dataset as side-effect, only called on the first cpu
    def prepare_data(self):
        MNIST("~/datasets", train=True, download=True)
        MNIST("~/datasets", train=False, download=True)

    # called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # Split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST("~/datasets", train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_train,
                [55000, 5000],
            )
        if stage == (None, "test"):
            self.mnist_test = MNIST(
                "~/datasets",
                train=False,
                transform=transform,
            )

    # Dataloaders
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test


# def train_on_mnist(batch_size=256) -> type:
#     class DataClass(pl.LightningModule):
#         datamodule = MNISTDataModule(batch_size=batch_size)

#         def __init__(self, *args, **kwargs):
#             prototype_initializer = kwargs.pop(
#                 "prototype_initializer", pt.components.Zeros((28, 28, 1)))
#             super().__init__(*args,
#                              prototype_initializer=prototype_initializer,
#                              **kwargs)

#     dc: Type[DataClass] = DataClass
#     return dc


# ABSTRACT
class GeneralDataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 32) -> None:
        super().__init__()
        self.train_dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)


# def train_on_dataset(dataset: Dataset, batch_size: int = 256):
#     class DataClass(pl.LightningModule):
#         datamodule = GeneralDataModule(dataset, batch_size)
#         datashape = dataset[0][0].shape
#         example_input_array = torch.zeros_like(dataset[0][0]).unsqueeze(0)

#         def __init__(self, *args: Any, **kwargs: Any) -> None:
#             prototype_initializer = kwargs.pop(
#                 "prototype_initializer",
#                 pt.components.Zeros(self.datashape),
#             )
#             super().__init__(*args,
#                              prototype_initializer=prototype_initializer,
#                              **kwargs)

#     return DataClass

# if __name__ == "__main__":
#     from prototorch.models import GLVQ

#     demo_dataset = pt.datasets.Iris()

#     TrainingClass: Type = train_on_dataset(demo_dataset)

#     class DemoGLVQ(TrainingClass, GLVQ):
#         """Model Definition."""

#     # Hyperparameters
#     hparams = dict(
#         distribution={
#             "num_classes": 3,
#             "prototypes_per_class": 4
#         },
#         lr=0.01,
#     )

#     initialized = DemoGLVQ(hparams)
#     print(initialized)
