"""Neural Gas example using the Iris dataset."""

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from prototorch.datasets.abstract import NumpyDataset
from prototorch.models.callbacks.visualization import VisNG2D
from prototorch.models.neural_gas import NeuralGas
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Dataset
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_ds = NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = DataLoader(train_ds, num_workers=0, batch_size=150)

    # Hyperparameters
    hparams = dict(
        input_dim=x_train.shape[1],
        num_prototypes=30,
        lr=0.01,
    )

    # Initialize the model
    model = NeuralGas(hparams)

    # Model summary
    print(model)

    # Callbacks
    vis = VisNG2D(x_train, y_train)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            vis,
        ],
    )

    # Training loop
    trainer.fit(model, train_loader)
