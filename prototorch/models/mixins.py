import pytorch_lightning as pl
import torch
from prototorch.core.components import Components


class ProtoTorchMixin(pl.LightningModule):
    """All mixins are ProtoTorchMixins."""


class NonGradientMixin(ProtoTorchMixin):
    """Mixin for custom non-gradient optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        raise NotImplementedError


class ImagePrototypesMixin(ProtoTorchMixin):
    """Mixin for models with image prototypes."""
    proto_layer: Components
    components: torch.Tensor

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Constrain the components to the range [0, 1] by clamping after updates."""
        self.proto_layer.components.data.clamp_(0.0, 1.0)

    def get_prototype_grid(self, num_columns=2, return_channels_last=True):
        from torchvision.utils import make_grid
        grid = make_grid(self.components, nrow=num_columns)
        if return_channels_last:
            grid = grid.permute((1, 2, 0))
        return grid.cpu()
