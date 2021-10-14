import matplotlib.pyplot as plt
import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch.core.initializers import SMCI, RandomNormalCompInitializer
from prototorch.models.expanded.clcc_glvq import GLVQ, GLVQhparams
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor

plt.gray()

if __name__ == "__main__":
    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)

    components_initializer = SMCI(train_ds)

    hparams = GLVQhparams(
        distribution=dict(
            num_classes=3,
            per_class=2,
        ),
        component_initializer=components_initializer,
    )
    model = GLVQ(hparams)

    print(model)
    # Callbacks
    vis = pt.models.VisGLVQ2D(data=train_ds)
    # Train
    trainer = pl.Trainer(callbacks=[vis], gpus=1)
    trainer.fit(model, train_loader)
