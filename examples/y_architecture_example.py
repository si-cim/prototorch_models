import prototorch as pt
import pytorch_lightning as pl
import torchmetrics
from prototorch.core import SMCI
from prototorch.models.y_arch.callbacks import (
    LogTorchmetricCallback,
    PlotLambdaMatrixToTensorboard,
    VisGMLVQ2D,
)
from prototorch.models.y_arch.library.gmlvq import GMLVQ
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

# ##############################################################################

if __name__ == "__main__":

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    # Dataset
    train_ds = pt.datasets.Iris()

    # Dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        num_workers=0,
        shuffle=True,
    )

    # ------------------------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------------------------

    # Select Initializer
    components_initializer = SMCI(train_ds)

    # Define Hyperparameters
    hyperparameters = GMLVQ.HyperParameters(
        lr=0.1,
        backbone_lr=5,
        input_dim=4,
        distribution=dict(
            num_classes=3,
            per_class=1,
        ),
        component_initializer=components_initializer,
    )

    # Create Model
    model = GMLVQ(hyperparameters)

    print(model)

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    # Controlling Callbacks
    stopping_criterion = LogTorchmetricCallback(
        'recall',
        torchmetrics.Recall,
        num_classes=3,
    )

    es = EarlyStopping(
        monitor=stopping_criterion.name,
        mode="max",
        patience=10,
    )

    # Visualization Callback
    vis = VisGMLVQ2D(data=train_ds)

    # Define trainer
    trainer = pl.Trainer(
        callbacks=[
            vis,
            stopping_criterion,
            es,
            PlotLambdaMatrixToTensorboard(),
        ],
        max_epochs=1000,
    )

    # Train
    trainer.fit(model, train_loader)
