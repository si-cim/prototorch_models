import pytorch_lightning as pl
import torch
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import euclidean_distance
from prototorch.functions.initializers import get_initializer
from prototorch.functions.losses import glvq_loss
from prototorch.modules.prototypes import Prototypes1D


class GLVQ(pl.LightningModule):
    """Generalized Learning Vector Quantization."""
    def __init__(self, lr=1e-3, **kwargs):
        super().__init__()
        self.lr = lr
        self.proto_layer = Prototypes1D(**kwargs)

    @property
    def prototypes(self):
        return self.proto_layer.prototypes.detach().numpy()

    @property
    def prototype_labels(self):
        return self.proto_layer.prototype_labels.detach().numpy()

    def forward(self, x):
        protos = self.proto_layer.prototypes
        dis = euclidean_distance(x, protos)
        return dis

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        dis = self(x)
        plabels = self.proto_layer.prototype_labels
        mu = glvq_loss(dis, y, prototype_labels=plabels)
        loss = mu.sum(dim=0)
        self.log("train_loss", loss)
        return loss

    def predict(self, x):
        with torch.no_grad():
            d = self(x)
            plabels = self.proto_layer.prototype_labels
            y_pred = wtac(d, plabels)
        return y_pred.numpy()


class ImageGLVQ(GLVQ):
    """GLVQ model that constrains the prototypes to the range [0, 1] by
    clamping after updates.
    """
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.proto_layer.prototypes.data.clamp_(0., 1.)
