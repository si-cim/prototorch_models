"""LVQ models that are optimized using non-gradient methods."""

import logging

from prototorch.core.losses import _get_dp_dm
from prototorch.nn.activations import get_activation
from prototorch.nn.wrappers import LambdaLayer

from .abstract import NonGradientMixin
from .glvq import GLVQ


class LVQ1(NonGradientMixin, GLVQ):
    """Learning Vector Quantization 1."""

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos, plables = self.proto_layer()
        x, y = train_batch
        dis = self.compute_distances(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            d = self.compute_distances(xi.view(1, -1))
            preds = self.competition_layer(d, plabels)
            w = d.argmin(1)
            if yi == preds:
                shift = xi - protos[w]
            else:
                shift = protos[w] - xi
            updated_protos = protos + 0.0
            updated_protos[w] = protos[w] + (self.hparams.lr * shift)
            self.proto_layer.load_state_dict({"_components": updated_protos},
                                             strict=False)

        logging.debug(f"dis={dis}")
        logging.debug(f"y={y}")
        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None


class LVQ21(NonGradientMixin, GLVQ):
    """Learning Vector Quantization 2.1."""

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos, plabels = self.proto_layer()

        x, y = train_batch
        dis = self.compute_distances(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            xi = xi.view(1, -1)
            yi = yi.view(1, )
            d = self.compute_distances(xi)
            (_, wp), (_, wn) = _get_dp_dm(d, yi, plabels, with_indices=True)
            shiftp = xi - protos[wp]
            shiftn = protos[wn] - xi
            updated_protos = protos + 0.0
            updated_protos[wp] = protos[wp] + (self.hparams.lr * shiftp)
            updated_protos[wn] = protos[wn] + (self.hparams.lr * shiftn)
            self.proto_layer.load_state_dict({"_components": updated_protos},
                                             strict=False)

        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None


class MedianLVQ(NonGradientMixin, GLVQ):
    """Median LVQ

    # TODO Avoid computing distances over and over

    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        self.transfer_layer = LambdaLayer(
            get_activation(self.hparams.transfer_fn))

    def _f(self, x, y, protos, plabels):
        d = self.distance_layer(x, protos)
        dp, dm = _get_dp_dm(d, y, plabels)
        mu = (dp - dm) / (dp + dm)
        invmu = -1.0 * mu
        f = self.transfer_layer(invmu, beta=self.hparams.transfer_beta) + 1.0
        return f

    def expectation(self, x, y, protos, plabels):
        f = self._f(x, y, protos, plabels)
        gamma = f / f.sum()
        return gamma

    def lower_bound(self, x, y, protos, plabels, gamma):
        f = self._f(x, y, protos, plabels)
        lower_bound = (gamma * f.log()).sum()
        return lower_bound

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos, plabels = self.proto_layer()

        x, y = train_batch
        dis = self.compute_distances(x)

        for i, _ in enumerate(protos):
            # Expectation step
            gamma = self.expectation(x, y, protos, plabels)
            lower_bound = self.lower_bound(x, y, protos, plabels, gamma)

            # Maximization step
            _protos = protos + 0
            for k, xk in enumerate(x):
                _protos[i] = xk
                _lower_bound = self.lower_bound(x, y, _protos, plabels, gamma)
                if _lower_bound > lower_bound:
                    logging.debug(f"Updating prototype {i} to data {k}...")
                    self.proto_layer.load_state_dict({"_components": _protos},
                                                     strict=False)
                    break

        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None
