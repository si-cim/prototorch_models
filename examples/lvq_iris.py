"""Classical LVQ using GLVQ example on the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Dataset
    from sklearn.datasets import load_iris
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=150)

    # Hyperparameters
    hparams = dict(
        nclasses=3,
        prototypes_per_class=2,
        prototype_initializer=pt.components.SMI(train_ds),
        #prototype_initializer=pt.components.Random(2),
        lr=0.005,
    )

    # Initialize the model
    model = pt.models.LVQ1(hparams)
    #model = pt.models.LVQ21(hparams)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=(x_train, y_train))

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[vis],
    )

    # Training loop
    trainer.fit(model, train_loader)
