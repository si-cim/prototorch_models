"""
Proto Y Architecture

Network architecture for Component based Learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import pytorch_lightning as pl
import torch
from torchmetrics import Metric


class Steps(enumerate):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"


class BaseYArchitecture(pl.LightningModule):

    @dataclass
    class HyperParameters:
        """
        Add all hyperparameters in the inherited class.
        """
        ...

    # Fields
    registered_metrics: dict[str, dict[type[Metric], Metric]] = {
        Steps.TRAINING: {},
        Steps.VALIDATION: {},
        Steps.TEST: {},
    }
    registered_metric_callbacks: dict[str, dict[type[Metric],
                                                set[Callable]]] = {
                                                    Steps.TRAINING: {},
                                                    Steps.VALIDATION: {},
                                                    Steps.TEST: {},
                                                }

    # Type Hints for Necessary Fields
    components_layer: torch.nn.Module

    def __init__(self, hparams) -> None:
        if isinstance(hparams, dict):
            self.save_hyperparameters(hparams)
            # TODO: => Move into Component Child
            del hparams["initialized_proto_shape"]
            hparams = self.HyperParameters(**hparams)
        else:
            hparams_dict = asdict(hparams)
            hparams_dict["component_initializer"] = None
            self.save_hyperparameters(hparams_dict, )

        super().__init__()

        # Common Steps
        self.init_components(hparams)
        self.init_backbone(hparams)
        self.init_comparison(hparams)
        self.init_competition(hparams)

        # Train Steps
        self.init_loss(hparams)

        # Inference Steps
        self.init_inference(hparams)

    # external API
    def get_competition(self, batch, components):
        '''
        Returns the output of the competition layer.
        '''
        latent_batch, latent_components = self.backbone(batch, components)
        # TODO: => Latent Hook
        comparison_tensor = self.comparison(latent_batch, latent_components)
        # TODO: => Comparison Hook
        return comparison_tensor

    def forward(self, batch):
        '''
        Returns the prediction.
        '''
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
        '''
        Returns the Output of the comparison layer.
        '''
        if isinstance(batch, torch.Tensor):
            batch = (batch, None)
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        return self.get_competition(batch, components)

    def loss_forward(self, batch):
        '''
        Returns the output of the loss layer.
        '''
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        comparison_tensor = self.get_competition(batch, components)
        # TODO: => Competition Hook
        return self.loss(comparison_tensor, batch, components)

    # Empty Initialization
    def init_components(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the components step.
        """

    def init_backbone(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the backbone step.
        """

    def init_comparison(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the comparison step.
        """

    def init_competition(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the competition step.
        """

    def init_loss(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the loss step.
        """

    def init_inference(self, hparams: HyperParameters) -> None:
        """
        All initialization necessary for the inference step.
        """

    # Empty Steps
    def components(self):
        """
        This step has no input.

        It returns the components.
        """
        raise NotImplementedError(
            "The components step has no reasonable default.")

    def backbone(self, batch, components):
        """
        The backbone step receives the data batch and the components.
        It can transform both by an arbitrary function.

        It returns the transformed batch and components,
        each of the same length as the original input.
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

    # Y Architecture Hooks

    # internal API, called by models and callbacks
    def register_torchmetric(
        self,
        name: Callable,
        metric: type[Metric],
        step: str = Steps.TRAINING,
        **metric_kwargs,
    ):
        '''
        Register a callback for evaluating a torchmetric.
        '''
        if step == Steps.PREDICT:
            raise ValueError("Prediction metrics are not supported.")

        if metric not in self.registered_metrics:
            self.registered_metrics[step][metric] = metric(**metric_kwargs)
            self.registered_metric_callbacks[step][metric] = {name}
        else:
            self.registered_metric_callbacks[step][metric].add(name)

    def update_metrics_step(self, batch, step):
        # Prediction Metrics
        preds = self(batch)

        _, y = batch
        for metric in self.registered_metrics[step]:
            instance = self.registered_metrics[step][metric].to(self.device)
            instance(y, preds.reshape(y.shape))

    def update_metrics_epoch(self, step):
        for metric in self.registered_metrics[step]:
            instance = self.registered_metrics[step][metric].to(self.device)
            value = instance.compute()

            for callback in self.registered_metric_callbacks[step][metric]:
                callback(value, self)

            instance.reset()

    # Lightning steps
    # -------------------------------------------------------------------------
    # >>>> Training
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.update_metrics_step(batch, Steps.TRAINING)

        return self.loss_forward(batch)

    def training_epoch_end(self, outputs) -> None:
        self.update_metrics_epoch(Steps.TRAINING)

    # >>>> Validation
    def validation_step(self, batch, batch_idx):
        self.update_metrics_step(batch, Steps.VALIDATION)

        return self.loss_forward(batch)

    def validation_epoch_end(self, outputs) -> None:
        self.update_metrics_epoch(Steps.VALIDATION)

    # >>>> Test
    def test_step(self, batch, batch_idx):
        self.update_metrics_step(batch, Steps.TEST)
        return self.loss_forward(batch)

    def test_epoch_end(self, outputs) -> None:
        self.update_metrics_epoch(Steps.TEST)

    # >>>> Prediction
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict(batch)

    # Check points
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Compatible with Lightning
        checkpoint["hyper_parameters"] = {
            'hparams': checkpoint["hyper_parameters"]
        }
        return super().on_save_checkpoint(checkpoint)
