"""Lightning Callbacks."""

import pytorch_lightning as pl
import torch


class PruneLoserPrototypes(pl.Callback):
    def __init__(self,
                 threshold=0.01,
                 prune_after_epochs=10,
                 prune_quota_per_epoch=-1,
                 frequency=1,
                 verbose=False):
        self.threshold = threshold  # minimum win ratio
        self.prune_after_epochs = prune_after_epochs  # epochs to wait
        self.prune_quota_per_epoch = prune_quota_per_epoch
        self.frequency = frequency
        self.verbose = verbose

    def on_epoch_start(self, trainer, pl_module):
        pl_module.initialize_prototype_win_ratios()

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) < self.prune_after_epochs:
            return None
        if (trainer.current_epoch + 1) % self.frequency:
            return None
        ratios = pl_module.prototype_win_ratios.mean(dim=0)
        to_prune = torch.arange(len(ratios))[ratios < self.threshold]
        if self.prune_quota_per_epoch > 0:
            to_prune = to_prune[:self.prune_quota_per_epoch]
        if len(to_prune) > 0:
            if self.verbose:
                print(f"\nPrototype win ratios: {ratios}")
                print(f"Pruning prototypes at: {to_prune.tolist()}")
            cur_num_protos = pl_module.num_prototypes
            pl_module.remove_prototypes(indices=to_prune)
            new_num_protos = pl_module.num_prototypes
            if self.verbose:
                print(f"`num_prototypes` reduced from {cur_num_protos} "
                      f"to {new_num_protos}.")
        return True
