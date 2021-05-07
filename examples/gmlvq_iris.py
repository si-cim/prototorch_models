"""GMLVQ example using all four dimensions of the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Dataset
    from sklearn.datasets import load_iris
    x_train, y_train = load_iris(return_X_y=True)
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)
    # Hyperparameters
    hparams = dict(
        nclasses=3,
        prototypes_per_class=1,
        input_dim=x_train.shape[1],
        latent_dim=x_train.shape[1],
        prototype_initializer=pt.components.SMI(train_ds),
        lr=0.01,
    )

    # Initialize the model
    model = pt.models.GMLVQ(hparams)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=100)

    # Training loop
    trainer.fit(model, train_loader)

    # Display the Lambda matrix
    model.show_lambda()
