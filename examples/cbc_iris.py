"""CBC example using the Iris dataset."""
import argparse
import warnings

import prototorch as pt
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from prototorch.models import CBC, VisCBC2D
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    # Dataset
    train_ds = pt.datasets.Iris(dims=[0, 2])

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=32)

    # Hyperparameters
    hparams = dict(
        distribution=[1, 0, 3],
        margin=0.1,
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = CBC(
        hparams,
        components_initializer=pt.initializers.SSCI(train_ds, noise=0.1),
        reasonings_initializer=pt.initializers.
        PurePositiveReasoningsInitializer(),
    )

    # Callbacks
    vis = VisCBC2D(
        data=train_ds,
        title="CBC Iris Example",
        resolution=100,
        axis_off=True,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else "auto",
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            vis,
        ],
        detect_anomaly=True,
        log_every_n_steps=1,
        max_epochs=1000,
    )

    # Training loop
    trainer.fit(model, train_loader)
