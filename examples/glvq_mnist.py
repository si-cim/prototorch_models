import pytorch_lightning as pl
import torchvision
from matplotlib import pyplot as plt
from prototorch.functions.initializers import stratified_mean
from prototorch.models.glvq import ImageGLVQ
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def plot_protos(protos, shape=(-1, 1, 28, 28), nrow=2):
    grid = torchvision.utils.make_grid(protos.reshape(*shape), nrow=nrow)
    grid = grid.permute((1, 2, 0))
    plt.imshow(grid)



if __name__ == "__main__":
    dataset = MNIST("./datasets",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=1024)
    val_loader = DataLoader(mnist_val, batch_size=1024)

    model = ImageGLVQ(input_dim=28 * 28, nclasses=10, prototypes_per_class=2)

    # Warm-start prototypes
    prototypes, prototype_labels = stratified_mean(
        x_train,
        y_train,
        prototype_distribution=self.prototype_distribution,
        one_hot=one_hot_labels,
    )

    trainer = pl.Trainer(gpus=0, max_epochs=3)

    trainer.fit(model, train_loader, val_loader)

    protos = model.proto_layer.prototypes.detach().cpu()
    plot_protos(protos, shape=(-1, 1, 28, 28), nrow=4)
    plt.show(block=True)
