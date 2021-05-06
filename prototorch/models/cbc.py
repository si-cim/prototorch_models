import pytorch_lightning as pl
import torch
import torchmetrics
from prototorch.components.components import Components
from prototorch.functions.distances import euclidean_distance
from prototorch.functions.similarities import cosine_similarity


def rescaled_cosine_similarity(x, y):
    """Cosine Similarity rescaled to [0, 1]."""
    similarities = cosine_similarity(x, y)
    return (similarities + 1.0) / 2.0


def shift_activation(x):
    return (x + 1.0) / 2.0


def euclidean_similarity(x, y):
    d = euclidean_distance(x, y)
    return torch.exp(-d * 3)


class CosineSimilarity(torch.nn.Module):
    def __init__(self, activation=shift_activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, y):
        epsilon = torch.finfo(x.dtype).eps
        normed_x = (x / x.pow(2).sum(dim=tuple(range(
            1, x.ndim)), keepdim=True).clamp(min=epsilon).sqrt()).flatten(
                start_dim=1)
        normed_y = (y / y.pow(2).sum(dim=tuple(range(
            1, y.ndim)), keepdim=True).clamp(min=epsilon).sqrt()).flatten(
                start_dim=1)
        # normed_x = (x / torch.linalg.norm(x, dim=1))
        diss = torch.inner(normed_x, normed_y)
        return self.activation(diss)


class MarginLoss(torch.nn.modules.loss._Loss):
    def __init__(self,
                 margin=0.3,
                 size_average=None,
                 reduce=None,
                 reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input_, target):
        dp = torch.sum(target * input_, dim=-1)
        dm = torch.max(input_ - target, dim=-1).values
        return torch.nn.functional.relu(dm - dp + self.margin)


class ReasoningLayer(torch.nn.Module):
    def __init__(self, n_components, n_classes, n_replicas=1):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_classes = n_classes
        probabilities_init = torch.zeros(2, 1, n_components, self.n_classes)
        probabilities_init.uniform_(0.4, 0.6)
        self.reasoning_probabilities = torch.nn.Parameter(probabilities_init)

    @property
    def reasonings(self):
        pk = self.reasoning_probabilities[0]
        nk = (1 - pk) * self.reasoning_probabilities[1]
        ik = 1 - pk - nk
        img = torch.cat([pk, nk, ik], dim=0).permute(1, 0, 2)
        return img.unsqueeze(1)

    def forward(self, detections):
        pk = self.reasoning_probabilities[0].clamp(0, 1)
        nk = (1 - pk) * self.reasoning_probabilities[1].clamp(0, 1)
        epsilon = torch.finfo(pk.dtype).eps
        numerator = (detections @ (pk - nk)) + nk.sum(1)
        probs = numerator / (pk + nk).sum(1)
        probs = probs.squeeze(0)
        return probs


class CBC(pl.LightningModule):
    """Classification-By-Components."""
    def __init__(self,
                 hparams,
                 margin=0.1,
                 backbone_class=torch.nn.Identity,
                 similarity=euclidean_similarity,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.margin = margin
        self.component_layer = Components(self.hparams.num_components,
                                          self.hparams.component_initializer)
        # self.similarity = CosineSimilarity()
        self.similarity = similarity
        self.backbone = backbone_class()
        self.backbone_dependent = backbone_class().requires_grad_(False)
        n_components = self.components.shape[0]
        self.reasoning_layer = ReasoningLayer(n_components=n_components,
                                              n_classes=self.hparams.nclasses)
        self.train_acc = torchmetrics.Accuracy()

    @property
    def components(self):
        return self.component_layer.components.detach().cpu()

    @property
    def reasonings(self):
        return self.reasoning_layer.reasonings.cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def sync_backbones(self):
        master_state = self.backbone.state_dict()
        self.backbone_dependent.load_state_dict(master_state, strict=True)

    def forward(self, x):
        self.sync_backbones()
        protos = self.component_layer()

        latent_x = self.backbone(x)
        latent_protos = self.backbone_dependent(protos)

        detections = self.similarity(latent_x, latent_protos)
        probs = self.reasoning_layer(detections)
        return probs

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_pred = self(x)
        nclasses = self.reasoning_layer.n_classes
        y_true = torch.nn.functional.one_hot(y.long(), num_classes=nclasses)
        loss = MarginLoss(self.margin)(y_pred, y_true).mean(dim=0)
        self.log("train_loss", loss)
        self.train_acc(y_pred, y_true)
        self.log(
            "acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict(self, x):
        with torch.no_grad():
            y_pred = self(x)
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred.numpy()


class ImageCBC(CBC):
    """CBC model that constrains the components to the range [0, 1] by
    clamping after updates.
    """
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self.component_layer.prototypes.data.clamp_(0.0, 1.0)
