import prototorch as pt
import pytorch_lightning as pl
import torchmetrics
from prototorch.core import SMCI
from prototorch.models.proto_y_architecture.callbacks import (
    LogTorchmetricCallback,
    VisGLVQ2D,
)
from prototorch.models.proto_y_architecture.glvq import GLVQ
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

# ##############################################################################

if __name__ == "__main__":

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])
    train_ds.targets[train_ds.targets == 2.0] = 1.0

    # Dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        num_workers=0,
        shuffle=True,
    )

    # ------------------------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------------------------

    # Select Initializer
    components_initializer = SMCI(train_ds)

    # Define Hyperparameters
    hyperparameters = GLVQ.HyperParameters(
        lr=0.5,
        distribution=dict(
            num_classes=2,
            per_class=1,
        ),
        component_initializer=components_initializer,
    )

    # Create Model
    model = GLVQ(hyperparameters)
    print(model)

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    # Controlling Callbacks
    stopping_criterion = LogTorchmetricCallback(
        'recall',
        torchmetrics.Recall,
        num_classes=2,
    )

    es = EarlyStopping(
        monitor=stopping_criterion.name,
        min_delta=0.001,
        patience=15,
        mode="max",
        check_on_train_epoch_end=True,
    )

    # Visualization Callback
    vis = VisGLVQ2D(data=train_ds)

    # Define trainer
    trainer = pl.Trainer(
        callbacks=[
            vis,
            stopping_criterion,
            es,
        ],
        gpus=0,
        max_epochs=200,
        log_every_n_steps=1,
    )

    # Train
    trainer.fit(model, train_loader)
