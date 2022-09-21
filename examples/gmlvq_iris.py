import logging

import pytorch_lightning as pl
import torchmetrics
from prototorch.core import SMCI, PCALinearTransformInitializer
from prototorch.datasets import Iris
from prototorch.models.architectures.base import Steps
from prototorch.models.callbacks import (
    LogTorchmetricCallback,
    PlotLambdaMatrixToTensorboard,
    VisGMLVQ2D,
)
from prototorch.models.library.gmlvq import GMLVQ
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

logging.basicConfig(level=logging.INFO)

# ##############################################################################


def main():
    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    # Dataset
    full_dataset = Iris()
    full_count = len(full_dataset)

    train_count = int(full_count * 0.5)
    val_count = int(full_count * 0.4)
    test_count = int(full_count * 0.1)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, (train_count, val_count, test_count))

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------------------------

    # Select Initializer
    components_initializer = SMCI(full_dataset)

    # Define Hyperparameters
    hyperparameters = GMLVQ.HyperParameters(
        lr=dict(components_layer=0.1, _omega=0),
        input_dim=4,
        distribution=dict(
            num_classes=3,
            per_class=1,
        ),
        component_initializer=components_initializer,
        omega_initializer=PCALinearTransformInitializer,
        omega_initializer_kwargs=dict(
            data=train_dataset.dataset[train_dataset.indices][0]))

    # Create Model
    model = GMLVQ(hyperparameters)

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    # Controlling Callbacks
    recall = LogTorchmetricCallback(
        'training_recall',
        torchmetrics.Recall,
        num_classes=3,
        step=Steps.TRAINING,
    )

    stopping_criterion = LogTorchmetricCallback(
        'validation_recall',
        torchmetrics.Recall,
        num_classes=3,
        step=Steps.VALIDATION,
    )

    accuracy = LogTorchmetricCallback(
        'validation_accuracy',
        torchmetrics.Accuracy,
        num_classes=3,
        step=Steps.VALIDATION,
    )

    es = EarlyStopping(
        monitor=stopping_criterion.name,
        mode="max",
        patience=10,
    )

    # Visualization Callback
    vis = VisGMLVQ2D(data=full_dataset)

    # Define trainer
    trainer = pl.Trainer(
        callbacks=[
            vis,
            recall,
            accuracy,
            stopping_criterion,
            es,
            PlotLambdaMatrixToTensorboard(),
        ],
        max_epochs=100,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Manual save
    trainer.save_checkpoint("./y_arch.ckpt")

    # Load saved model
    new_model = GMLVQ.load_from_checkpoint(
        checkpoint_path="./y_arch.ckpt",
        strict=True,
    )


if __name__ == "__main__":
    main()
