import prototorch as pt
import pytorch_lightning as pl
import torchmetrics
from prototorch.core import SMCI
from prototorch.y.callbacks import (
    LogTorchmetricCallback,
    PlotLambdaMatrixToTensorboard,
    VisGMLVQ2D,
)
from prototorch.y.library.gmlvq import GMLVQ
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

# ##############################################################################


def main():
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
        lr=dict(components_layer=0.1, _omega=0),
        input_dim=4,
        distribution=dict(
            num_classes=3,
            per_class=1,
        ),
        component_initializer=components_initializer,
    )

    # Create Model
    model = GMLVQ(hyperparameters)

    print(model.hparams)

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
    trainer = pl.Trainer(callbacks=[
        vis,
        stopping_criterion,
        es,
        PlotLambdaMatrixToTensorboard(),
    ], )

    # Train
    trainer.fit(model, train_loader)

    # Manual save
    trainer.save_checkpoint("./y_arch.ckpt")

    # Load saved model
    new_model = GMLVQ.load_from_checkpoint(
        checkpoint_path="./y_arch.ckpt",
        strict=True,
    )

    print(new_model.hparams)


if __name__ == "__main__":
    main()
