"""
CLCC Scheme

CLCC is a LVQ scheme containing 4 steps
- Components
- Latent Space
- Comparison
- Competition

"""
from dataclasses import dataclass
from typing import (
    Dict,
    Set,
    Type,
)

import pytorch_lightning as pl
import torch
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy


class BaseYArchitecture(pl.LightningModule):

    @dataclass
    class HyperParameters:
        ...

    registered_metrics: Dict[Type[Metric], Metric] = {}
    registered_metric_names: Dict[Type[Metric], Set[str]] = {}

    components_layer: torch.nn.Module

    def __init__(self, hparams) -> None:
        super().__init__()

        # Common Steps
        self.init_components(hparams)
        self.init_latent(hparams)
        self.init_comparison(hparams)
        self.init_competition(hparams)

        # Train Steps
        self.init_loss(hparams)

        # Inference Steps
        self.init_inference(hparams)

        # Initialize Model Metrics
        self.init_model_metrics()

    # internal API, called by models and callbacks
    def register_torchmetric(
        self,
        name: str,
        metric: Type[Metric],
        **metric_kwargs,
    ):
        if metric not in self.registered_metrics:
            self.registered_metrics[metric] = metric(**metric_kwargs)
            self.registered_metric_names[metric] = {name}
        else:
            self.registered_metric_names[metric].add(name)

    # external API
    def get_competition(self, batch, components):
        latent_batch, latent_components = self.latent(batch, components)
        # TODO: => Latent Hook
        comparison_tensor = self.comparison(latent_batch, latent_components)
        # TODO: => Comparison Hook
        return comparison_tensor

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = (batch, None)
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        comparison_tensor = self.get_competition(batch, components)
        # TODO: => Competition Hook
        return self.inference(comparison_tensor, components)

    def predict(self, batch):
        """
        Alias for forward
        """
        return self.forward(batch)

    def forward_comparison(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = (batch, None)
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        return self.get_competition(batch, components)

    def loss_forward(self, batch):
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        comparison_tensor = self.get_competition(batch, components)
        # TODO: => Competition Hook
        return self.loss(comparison_tensor, batch, components)

    # Empty Initialization
    # TODO: Type hints
    # TODO: Docs
    def init_components(self, hparams: HyperParameters) -> None:
        ...

    def init_latent(self, hparams: HyperParameters) -> None:
        ...

    def init_comparison(self, hparams: HyperParameters) -> None:
        ...

    def init_competition(self, hparams: HyperParameters) -> None:
        ...

    def init_loss(self, hparams: HyperParameters) -> None:
        ...

    def init_inference(self, hparams: HyperParameters) -> None:
        ...

    def init_model_metrics(self) -> None:
        self.register_torchmetric('accuracy', Accuracy)

    # Empty Steps
    # TODO: Type hints
    def components(self):
        """
        This step has no input.

        It returns the components.
        """
        raise NotImplementedError(
            "The components step has no reasonable default.")

    def latent(self, batch, components):
        """
        The latent step receives the data batch and the components.
        It can transform both by an arbitrary function.

        It returns the transformed batch and components, each of the same length as the original input.
        """
        return batch, components

    def comparison(self, batch, components):
        """
        Takes a batch of size N and the component set of size M.

        It returns an NxMxD tensor containing D (usually 1) pairwise comparison measures.
        """
        raise NotImplementedError(
            "The comparison step has no reasonable default.")

    def competition(self, comparison_measures, components):
        """
        Takes the tensor of comparison measures.

        Assigns a competition vector to each class.
        """
        raise NotImplementedError(
            "The competition step has no reasonable default.")

    def loss(self, comparison_measures, batch, components):
        """
        Takes the tensor of competition measures.

        Calculates a single loss value
        """
        raise NotImplementedError("The loss step has no reasonable default.")

    def inference(self, comparison_measures, components):
        """
        Takes the tensor of competition measures.

        Returns the inferred vector.
        """
        raise NotImplementedError(
            "The inference step has no reasonable default.")

    def update_metrics_step(self, batch):
        x, y = batch

        # Prediction Metrics
        preds = self(x)
        for metric in self.registered_metrics:
            instance = self.registered_metrics[metric].to(self.device)
            instance(y, preds)

    def update_metrics_epoch(self):
        for metric in self.registered_metrics:
            instance = self.registered_metrics[metric].to(self.device)
            value = instance.compute()

            for name in self.registered_metric_names[metric]:
                self.log(name, value)

            instance.reset()

    # Lightning Hooks
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.update_metrics_step(batch)

        return self.loss_forward(batch)

    def training_epoch_end(self, outs) -> None:
        self.update_metrics_epoch()

    def validation_step(self, batch, batch_idx):
        return self.loss_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.loss_forward(batch)
