"""CBC example using the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Dataset
    from sklearn.datasets import load_iris
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Reproducibility
    pl.utilities.seed.seed_everything(seed=2)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)

    # Hyperparameters
    hparams = dict(
        input_dim=x_train.shape[1],
        nclasses=3,
        num_components=5,
        component_initializer=pt.components.SSI(train_ds, noise=0.01),
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.CBC(hparams)

    # Callbacks
    dvis = pt.models.VisCBC2D(data=(x_train, y_train),
                              title="CBC Iris Example")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[
            dvis,
        ],
    )

    # Training loop
    trainer.fit(model, train_loader)
