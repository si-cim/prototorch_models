"""ProtoTorch Neural Additive Model."""

import torch
import torchmetrics

from .abstract import ProtoTorchBolt


class BinaryNAM(ProtoTorchBolt):
    """Neural Additive Model for binary classification.

    Paper: https://arxiv.org/abs/2004.13912
    Official implementation: https://github.com/google-research/google-research/tree/master/neural_additive_models

    """
    def __init__(self, hparams: dict, extractors: torch.nn.ModuleList,
                 **kwargs):
        super().__init__(hparams, **kwargs)

        # Default hparams
        self.hparams.setdefault("threshold", 0.5)

        self.extractors = extractors
        self.linear = torch.nn.Linear(in_features=len(extractors),
                                      out_features=1,
                                      bias=True)

    def extract(self, x):
        """Apply the local extractors batch-wise on features."""
        out = torch.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = self.extractors[j](x[:, j].unsqueeze(1)).squeeze()
        return out

    def forward(self, x):
        x = self.extract(x)
        x = self.linear(x)
        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        preds = self(x).squeeze()
        train_loss = torch.nn.functional.binary_cross_entropy(preds, y.float())
        self.log("train_loss", train_loss)
        accuracy = torchmetrics.functional.accuracy(preds.int(), y.int())
        self.log("train_acc",
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return train_loss

    def predict(self, x):
        out = self(x)
        pred = torch.zeros_like(out, device=self.device)
        pred[out > self.hparams.threshold] = 1
        return pred
