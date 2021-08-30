"""Lightning Callbacks."""

import logging

import pytorch_lightning as pl
import torch

from ..core.components import Components
from ..core.initializers import LiteralCompInitializer
from .extras import ConnectionTopology


class PruneLoserPrototypes(pl.Callback):
    def __init__(self,
                 threshold=0.01,
                 idle_epochs=10,
                 prune_quota_per_epoch=-1,
                 frequency=1,
                 replace=False,
                 prototypes_initializer=None,
                 verbose=False):
        self.threshold = threshold  # minimum win ratio
        self.idle_epochs = idle_epochs  # epochs to wait before pruning
        self.prune_quota_per_epoch = prune_quota_per_epoch
        self.frequency = frequency
        self.replace = replace
        self.verbose = verbose
        self.prototypes_initializer = prototypes_initializer

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) < self.idle_epochs:
            return None
        if (trainer.current_epoch + 1) % self.frequency:
            return None

        ratios = pl_module.prototype_win_ratios.mean(dim=0)
        to_prune = torch.arange(len(ratios))[ratios < self.threshold]
        to_prune = to_prune.tolist()
        prune_labels = pl_module.prototype_labels[to_prune]
        if self.prune_quota_per_epoch > 0:
            to_prune = to_prune[:self.prune_quota_per_epoch]
            prune_labels = prune_labels[:self.prune_quota_per_epoch]

        if len(to_prune) > 0:
            if self.verbose:
                print(f"\nPrototype win ratios: {ratios}")
                print(f"Pruning prototypes at: {to_prune}")
                print(f"Corresponding labels are: {prune_labels.tolist()}")
            cur_num_protos = pl_module.num_prototypes
            pl_module.remove_prototypes(indices=to_prune)
            if self.replace:
                labels, counts = torch.unique(prune_labels,
                                              sorted=True,
                                              return_counts=True)
                distribution = dict(zip(labels.tolist(), counts.tolist()))
                if self.verbose:
                    print(f"Re-adding pruned prototypes...")
                    print(f"distribution={distribution}")
                pl_module.add_prototypes(
                    distribution=distribution,
                    components_initializer=self.prototypes_initializer)
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


class GNGCallback(pl.Callback):
    """GNG Callback.

    Applies growing algorithm based on accumulated error and topology.

    Based on "A Growing Neural Gas Network Learns Topologies" by Bernd Fritzke.

    """
    def __init__(self, reduction=0.1, freq=10):
        self.reduction = reduction
        self.freq = freq

    def on_epoch_end(self, trainer: pl.Trainer, pl_module):
        if (trainer.current_epoch + 1) % self.freq == 0:
            # Get information
            errors = pl_module.errors
            topology: ConnectionTopology = pl_module.topology_layer
            components: Components = pl_module.proto_layer.components

            # Insertion point
            worst = torch.argmax(errors)

            neighbors = topology.get_neighbors(worst)[0]

            if len(neighbors) == 0:
                logging.log(level=20, msg="No neighbor-pairs found!")
                return

            neighbors_errors = errors[neighbors]
            worst_neighbor = neighbors[torch.argmax(neighbors_errors)]

            # New Prototype
            new_component = 0.5 * (components[worst] +
                                   components[worst_neighbor])

            # Add component
            pl_module.proto_layer.add_components(
                None,
                initializer=LiteralCompInitializer(new_component.unsqueeze(0)))

            # Adjust Topology
            topology.add_prototype()
            topology.add_connection(worst, -1)
            topology.add_connection(worst_neighbor, -1)
            topology.remove_connection(worst, worst_neighbor)

            # New errors
            worst_error = errors[worst].unsqueeze(0)
            pl_module.errors = torch.cat([pl_module.errors, worst_error])
            pl_module.errors[worst] = errors[worst] * self.reduction
            pl_module.errors[
                worst_neighbor] = errors[worst_neighbor] * self.reduction

            trainer.accelerator.setup_optimizers(trainer)
