"""Callbacks for Pytorch Lighning Modules"""

import pytorch_lightning as pl
import torch


class StopOnNaN(pl.Callback):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def on_epoch_end(self, trainer, pl_module, logs={}):
        if torch.isnan(self.param).any():
            raise ValueError("NaN encountered. Stopping.")
