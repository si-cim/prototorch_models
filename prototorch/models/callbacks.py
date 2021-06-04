"""Lightning Callbacks."""

import pytorch_lightning as pl
import torch


class PruneLoserPrototypes(pl.Callback):
    def __init__(self,
                 threshold=0.01,
                 idle_epochs=10,
                 prune_quota_per_epoch=-1,
                 frequency=1,
                 replace=False,
                 initializer=None,
                 verbose=False):
        self.threshold = threshold  # minimum win ratio
        self.idle_epochs = idle_epochs  # epochs to wait before pruning
        self.prune_quota_per_epoch = prune_quota_per_epoch
        self.frequency = frequency
        self.replace = replace
        self.verbose = verbose
        self.initializer = initializer

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) < self.idle_epochs:
            return None
        if (trainer.current_epoch + 1) % self.frequency:
            return None
        ratios = pl_module.prototype_win_ratios.mean(dim=0)
        to_prune = torch.arange(len(ratios))[ratios < self.threshold]
        prune_labels = pl_module.prototype_labels[to_prune.tolist()]
        if self.prune_quota_per_epoch > 0:
            to_prune = to_prune[:self.prune_quota_per_epoch]
            prune_labels = prune_labels[:self.prune_quota_per_epoch]
        if len(to_prune) > 0:
            if self.verbose:
                print(f"\nPrototype win ratios: {ratios}")
                print(f"Pruning prototypes at: {to_prune.tolist()}")
            cur_num_protos = pl_module.num_prototypes
            pl_module.remove_prototypes(indices=to_prune)
            if self.replace:
                if self.verbose:
                    print(f"Re-adding prototypes at: {to_prune.tolist()}")
                labels, counts = torch.unique(prune_labels,
                                              sorted=True,
                                              return_counts=True)
                distribution = dict(zip(labels.tolist(), counts.tolist()))
                print(f"{distribution=}")
                pl_module.add_prototypes(distribution=distribution,
                                         initializer=self.initializer)
            new_num_protos = pl_module.num_prototypes
            if self.verbose:
                print(f"`num_prototypes` changed from {cur_num_protos} "
                      f"to {new_num_protos}.")
        return True


class PrototypeConvergence(pl.Callback):
    def __init__(self, min_delta=0.01, idle_epochs=10, verbose=False):
        self.min_delta = min_delta
        self.idle_epochs = idle_epochs  # epochs to wait
        self.verbose = verbose

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) < self.idle_epochs:
            return None
        if self.verbose:
            print("Stopping...")
        # TODO
        return True
