"""LVQ models that are optimized using non-gradient methods."""

from prototorch.functions.competitions import wtac
from prototorch.functions.losses import _get_dp_dm

from .glvq import GLVQ


class NonGradientLVQ(GLVQ):
    """Abstract Model for Models that do not use gradients in their update phase."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        raise NotImplementedError


class LVQ1(NonGradientLVQ):
    """Learning Vector Quantization 1."""
    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos = self.proto_layer.components
        plabels = self.proto_layer.component_labels

        x, y = train_batch
        dis = self._forward(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            d = self._forward(xi.view(1, -1))
            preds = wtac(d, plabels)
            w = d.argmin(1)
            if yi == preds:
                shift = xi - protos[w]
            else:
                shift = protos[w] - xi
            updated_protos = protos + 0.0
            updated_protos[w] = protos[w] + (self.hparams.lr * shift)
            self.proto_layer.load_state_dict({"_components": updated_protos},
                                             strict=False)

        # Logging
        self.log_acc(dis, y, tag="train_acc")

        return None


class LVQ21(NonGradientLVQ):
    """Learning Vector Quantization 2.1."""
    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        protos = self.proto_layer.components
        plabels = self.proto_layer.component_labels

        x, y = train_batch
        dis = self._forward(x)
        # TODO Vectorized implementation

        for xi, yi in zip(x, y):
            xi = xi.view(1, -1)
            yi = yi.view(1, )
            d = self._forward(xi)
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


class MedianLVQ(NonGradientLVQ):
    """Median LVQ"""
