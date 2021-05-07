"""Neural Gas example using the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Prepare and pre-process the dataset
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)

    # Hyperparameters
    hparams = dict(num_prototypes=30, lr=0.03)

    # Initialize the model
    model = pt.models.NeuralGas(hparams)

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisNG2D(data=train_ds)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=200, callbacks=[vis])

    # Training loop
    trainer.fit(model, train_loader)
