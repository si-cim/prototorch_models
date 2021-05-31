import prototorch as pt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


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


def train_on_mnist(batch_size=256) -> type:
    class DataClass(pl.LightningModule):
        datamodule = MNISTDataModule(batch_size=batch_size)

        def prototype_initializer(self, **kwargs):
            return pt.components.Zeros((28, 28, 1))

    return DataClass