"""LVQ1 example using the Iris dataset."""

import prototorch as pt
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # Acquire data
    from sklearn.datasets import load_iris
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[:, [0, 2]]

    # Relabel classes
    # y_train[y_train == 0] = 3
    # y_train[y_train == 1] = 4
    # y_train[y_train == 2] = 6

    # Dataset
    train_ds = pt.datasets.NumpyDataset(x_train, y_train)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True)

    # Hyperparameters
    num_classes = 3
    prototypes_per_class = 10
    hparams = dict(
        distribution={
            # class_label: num_prototypes
            # 3: 1,
            # 4: 2,
            # 6: 3,
            0: 1,
            2: 2,
            3: 3,
        },
        lr=0.001,
    )

    # Initialize the model
    model = pt.models.LVQ1(
        hparams,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
    )

    # Check if `num_classes` is correct
    print(f"{model.num_classes=}")
    assert model.num_classes == 3

    # Compute intermediate input and output sizes
    model.example_input_array = torch.zeros(4, 2)

    # Model summary
    print(model)

    # Callbacks
    vis = pt.models.VisGLVQ2D(data=(x_train, y_train),
                              cmap="viridis",
                              resolution=200,
                              block=False)

    # Setup trainer
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=50,
        callbacks=[vis],
        # fast_dev_run=1,
    )

    # Get prototype labels
    print(f"Protoype Labels are: ", model.prototype_labels.tolist())

    # Training loop
    trainer.fit(model, train_loader)
