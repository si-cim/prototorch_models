"""Unsupervised prototype learning algorithms."""

import numpy as np
import torch

from ..core.competitions import wtac
from ..core.distances import squared_euclidean_distance
from ..core.losses import NeuralGasEnergy
from ..nn.wrappers import LambdaLayer
from .abstract import NonGradientMixin, UnsupervisedPrototypeModel
from .callbacks import GNGCallback
from .extras import ConnectionTopology


class KohonenSOM(NonGradientMixin, UnsupervisedPrototypeModel):
    """Kohonen Self-Organizing-Map.

    TODO Allow non-2D grids

    """
    def __init__(self, hparams, **kwargs):
        h, w = hparams.get("shape")
        # Ignore `num_prototypes`
        hparams["num_prototypes"] = h * w
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Hyperparameters
        self.save_hyperparameters(hparams)

        # Default hparams
        self.hparams.setdefault("alpha", 0.3)
        self.hparams.setdefault("sigma", max(h, w) / 2.0)

        # Additional parameters
        x, y = torch.arange(h), torch.arange(w)
        grid = torch.stack(torch.meshgrid(x, y), dim=-1)
        self.register_buffer("_grid", grid)
        self._sigma = self.hparams.sigma
        self._lr = self.hparams.lr

    def predict_from_distances(self, distances):
        grid = self._grid.view(-1, 2)
        wp = wtac(distances, grid)
        return wp

    def training_step(self, train_batch, batch_idx):
        # x = train_batch
        # TODO Check if the batch has labels
        x = train_batch[0]
        d = self.compute_distances(x)
        wp = self.predict_from_distances(d)
        grid = self._grid.view(-1, 2)
        gd = squared_euclidean_distance(wp, grid)
        nh = torch.exp(-gd / self._sigma**2)
        protos = self.proto_layer()
        diff = x.unsqueeze(dim=1) - protos
        delta = self._lr * self.hparams.alpha * nh.unsqueeze(-1) * diff
        updated_protos = protos + delta.sum(dim=0)
        self.proto_layer.load_state_dict({"_components": updated_protos},
                                         strict=False)

    def training_epoch_end(self, training_step_outputs):
        self._sigma = self.hparams.sigma * np.exp(
            -self.current_epoch / self.trainer.max_epochs)

    def extra_repr(self):
        return f"(grid): (shape: {tuple(self._grid.shape)})"


class HeskesSOM(UnsupervisedPrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

    def training_step(self, train_batch, batch_idx):
        # TODO Implement me!
        raise NotImplementedError()


class NeuralGas(UnsupervisedPrototypeModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Hyperparameters
        self.save_hyperparameters(hparams)

        # Default hparams
        self.hparams.setdefault("agelimit", 10)
        self.hparams.setdefault("lm", 1)

        self.energy_layer = NeuralGasEnergy(lm=self.hparams.lm)
        self.topology_layer = ConnectionTopology(
            agelimit=self.hparams.agelimit,
            num_prototypes=self.hparams.num_prototypes,
        )

    def training_step(self, train_batch, batch_idx):
        # x = train_batch
        # TODO Check if the batch has labels
        x = train_batch[0]
        d = self.compute_distances(x)
        loss, _ = self.energy_layer(d)
        self.topology_layer(d)
        self.log("loss", loss)
        return loss

    # def training_epoch_end(self, training_step_outputs):
    #     print(f"{self.trainer.lr_schedulers}")
    #     print(f"{self.trainer.lr_schedulers[0]['scheduler'].optimizer}")


class GrowingNeuralGas(NeuralGas):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Defaults
        self.hparams.setdefault("step_reduction", 0.5)
        self.hparams.setdefault("insert_reduction", 0.1)
        self.hparams.setdefault("insert_freq", 10)

        errors = torch.zeros(self.hparams.num_prototypes, device=self.device)
        self.register_buffer("errors", errors)

    def training_step(self, train_batch, _batch_idx):
        # x = train_batch
        # TODO Check if the batch has labels
        x = train_batch[0]
        d = self.compute_distances(x)
        loss, order = self.energy_layer(d)
        winner = order[:, 0]
        mask = torch.zeros_like(d)
        mask[torch.arange(len(mask)), winner] = 1.0
        dp = d * mask

        self.errors += torch.sum(dp * dp)
        self.errors *= self.hparams.step_reduction

        self.topology_layer(d)
        self.log("loss", loss)
        return loss

    def configure_callbacks(self):
        return [
            GNGCallback(reduction=self.hparams.insert_reduction,
                        freq=self.hparams.insert_freq)
        ]
