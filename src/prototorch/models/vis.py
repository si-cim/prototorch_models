"""Visualization Callbacks."""

import warnings
from typing import Sized

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from matplotlib import pyplot as plt
from prototorch.utils.colors import get_colors, get_legend_handles
from prototorch.utils.utils import mesh2d
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset


class Vis2DAbstract(pl.Callback):

    def __init__(self,
                 data=None,
                 title="Prototype Visualization",
                 cmap="viridis",
                 xlabel="Data dimension 1",
                 ylabel="Data dimension 2",
                 legend_labels=None,
                 border=0.1,
                 resolution=100,
                 flatten_data=True,
                 axis_off=False,
                 show_protos=True,
                 show=True,
                 tensorboard=False,
                 show_last_only=False,
                 pause_time=0.1,
                 block=False):
        super().__init__()

        if data:
            if isinstance(data, Dataset):
                if isinstance(data, Sized):
                    x, y = next(iter(DataLoader(data, batch_size=len(data))))
                else:
                    # TODO: Add support for non-sized datasets
                    raise NotImplementedError(
                        "Data must be a dataset with a __len__ method.")
            elif isinstance(data, DataLoader):
                x = torch.tensor([])
                y = torch.tensor([])
                for x_b, y_b in data:
                    x = torch.cat([x, x_b])
                    y = torch.cat([y, y_b])
            else:
                x, y = data

            if flatten_data:
                x = x.reshape(len(x), -1)

            self.x_train = x
            self.y_train = y
        else:
            self.x_train = None
            self.y_train = None

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend_labels = legend_labels
        self.fig = plt.figure(self.title)
        self.cmap = cmap
        self.border = border
        self.resolution = resolution
        self.axis_off = axis_off
        self.show_protos = show_protos
        self.show = show
        self.tensorboard = tensorboard
        self.show_last_only = show_last_only
        self.pause_time = pause_time
        self.block = block

    def precheck(self, trainer):
        if self.show_last_only:
            if trainer.current_epoch != trainer.max_epochs - 1:
                return False
        return True

    def setup_ax(self):
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.axis_off:
            ax.axis("off")
        return ax

    def plot_data(self, ax, x, y):
        ax.scatter(
            x[:, 0],
            x[:, 1],
            c=y,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )

    def plot_protos(self, ax, protos, plabels):
        ax.scatter(
            protos[:, 0],
            protos[:, 1],
            c=plabels,
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )

    def add_to_tensorboard(self, trainer, pl_module):
        tb = pl_module.logger.experiment
        tb.add_figure(tag=f"{self.title}",
                      figure=self.fig,
                      global_step=trainer.current_epoch,
                      close=False)

    def log_and_display(self, trainer, pl_module):
        if self.tensorboard:
            self.add_to_tensorboard(trainer, pl_module)
        if self.show:
            if not self.block:
                plt.pause(self.pause_time)
            else:
                plt.show(block=self.block)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True
        self.visualize(pl_module)
        self.log_and_display(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        plt.close()

    def visualize(self, pl_module):
        raise NotImplementedError


class VisGLVQ2D(Vis2DAbstract):

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        ax = self.setup_ax()
        self.plot_protos(ax, protos, plabels)
        if x_train is not None:
            self.plot_data(ax, x_train, y_train)
            mesh_input, xx, yy = mesh2d(np.vstack([x_train, protos]),
                                        self.border, self.resolution)
        else:
            mesh_input, xx, yy = mesh2d(protos, self.border, self.resolution)
        _components = pl_module.proto_layer._components
        mesh_input = torch.from_numpy(mesh_input).type_as(_components)
        y_pred = pl_module.predict(mesh_input)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)


class VisSiameseGLVQ2D(Vis2DAbstract):

    def __init__(self, *args, map_protos=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_protos = map_protos

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        device = pl_module.device
        with torch.no_grad():
            x_train = pl_module.backbone(torch.Tensor(x_train).to(device))
            x_train = x_train.cpu().detach()
        if self.map_protos:
            with torch.no_grad():
                protos = pl_module.backbone(torch.Tensor(protos).to(device))
                protos = protos.cpu().detach()
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        if self.show_protos:
            self.plot_protos(ax, protos, plabels)
            x = np.vstack((x_train, protos))
            mesh_input, xx, yy = mesh2d(x, self.border, self.resolution)
        else:
            mesh_input, xx, yy = mesh2d(x_train, self.border, self.resolution)
        _components = pl_module.proto_layer._components
        mesh_input = torch.Tensor(mesh_input).type_as(_components)
        y_pred = pl_module.predict_latent(mesh_input,
                                          map_protos=self.map_protos)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)


class VisGMLVQ2D(Vis2DAbstract):

    def __init__(self, *args, ev_proj=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_proj = ev_proj

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        device = pl_module.device
        omega = pl_module._omega.detach()
        lam = omega @ omega.T
        u, _, _ = torch.pca_lowrank(lam, q=2)
        with torch.no_grad():
            x_train = torch.Tensor(x_train).to(device)
            x_train = x_train @ u
            x_train = x_train.cpu().detach()
        if self.show_protos:
            with torch.no_grad():
                protos = torch.Tensor(protos).to(device)
                protos = protos @ u
                protos = protos.cpu().detach()
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        if self.show_protos:
            self.plot_protos(ax, protos, plabels)


class VisCBC2D(Vis2DAbstract):

    def visualize(self, pl_module):
        x_train, y_train = self.x_train, self.y_train
        protos = pl_module.components
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        self.plot_protos(ax, protos, "w")
        x = np.vstack((x_train, protos))
        mesh_input, xx, yy = mesh2d(x, self.border, self.resolution)
        _components = pl_module.components_layer._components
        y_pred = pl_module.predict(
            torch.Tensor(mesh_input).type_as(_components))
        y_pred = y_pred.cpu().reshape(xx.shape)

        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)


class VisNG2D(Vis2DAbstract):

    def visualize(self, pl_module):
        x_train, y_train = self.x_train, self.y_train
        protos = pl_module.prototypes
        cmat = pl_module.topology_layer.cmat.cpu().numpy()

        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        self.plot_protos(ax, protos, "w")

        # Draw connections
        for i in range(len(protos)):
            for j in range(i, len(protos)):
                if cmat[i][j]:
                    ax.plot(
                        [protos[i, 0], protos[j, 0]],
                        [protos[i, 1], protos[j, 1]],
                        "k-",
                    )


class VisSpectralProtos(Vis2DAbstract):

    def visualize(self, pl_module):
        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        ax = self.setup_ax()
        colors = get_colors(vmax=max(plabels), vmin=min(plabels))
        for p, pl in zip(protos, plabels):
            ax.plot(p, c=colors[int(pl)])
        if self.legend_labels:
            handles = get_legend_handles(
                colors,
                self.legend_labels,
                marker="lines",
            )
            ax.legend(handles=handles)


class VisImgComp(Vis2DAbstract):

    def __init__(self,
                 *args,
                 random_data=0,
                 dataformats="CHW",
                 num_columns=2,
                 add_embedding=False,
                 embedding_data=100,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.random_data = random_data
        self.dataformats = dataformats
        self.num_columns = num_columns
        self.add_embedding = add_embedding
        self.embedding_data = embedding_data

    def on_train_start(self, _, pl_module):
        if isinstance(pl_module.logger, TensorBoardLogger):
            tb = pl_module.logger.experiment

            # Add embedding
            if self.add_embedding:
                if self.x_train is not None and self.y_train is not None:
                    ind = np.random.choice(len(self.x_train),
                                           size=self.embedding_data,
                                           replace=False)
                    data = self.x_train[ind]
                    tb.add_embedding(data.view(len(ind), -1),
                                     label_img=data,
                                     global_step=None,
                                     tag="Data Embedding",
                                     metadata=self.y_train[ind],
                                     metadata_header=None)
                else:
                    raise ValueError("No data for add embedding flag")

            # Random Data
            if self.random_data:
                if self.x_train is not None:
                    ind = np.random.choice(len(self.x_train),
                                           size=self.random_data,
                                           replace=False)
                    data = self.x_train[ind]
                    grid = torchvision.utils.make_grid(data,
                                                       nrow=self.num_columns)
                    tb.add_image(tag="Data",
                                 img_tensor=grid,
                                 global_step=None,
                                 dataformats=self.dataformats)
                else:
                    raise ValueError("No data for random data flag")

        else:
            warnings.warn(
                f"TensorBoardLogger is required, got {type(pl_module.logger)}")

    def add_to_tensorboard(self, trainer, pl_module):
        tb = pl_module.logger.experiment

        components = pl_module.components
        grid = torchvision.utils.make_grid(components, nrow=self.num_columns)
        tb.add_image(
            tag="Components",
            img_tensor=grid,
            global_step=trainer.current_epoch,
            dataformats=self.dataformats,
        )

    def visualize(self, pl_module):
        if self.show:
            components = pl_module.components
            grid = torchvision.utils.make_grid(components,
                                               nrow=self.num_columns)
            plt.imshow(grid.permute((1, 2, 0)).cpu(), cmap=self.cmap)
