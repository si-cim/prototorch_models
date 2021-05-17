import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from prototorch.utils.celluloid import Camera
from prototorch.utils.colors import color_scheme
from prototorch.utils.utils import (gif_from_dir, make_directory,
                                    prettify_string)
from torch.utils.data import DataLoader, Dataset


class Vis2DAbstract(pl.Callback):
    def __init__(self,
                 data,
                 title="Prototype Visualization",
                 cmap="viridis",
                 border=0.1,
                 resolution=100,
                 axis_off=False,
                 show_protos=True,
                 show=True,
                 tensorboard=False,
                 show_last_only=False,
                 pause_time=0.1,
                 block=False):
        super().__init__()

        if isinstance(data, Dataset):
            x, y = next(iter(DataLoader(data, batch_size=len(data))))
            x = x.view(len(data), -1)  # flatten
        else:
            x, y = data
        self.x_train = x
        self.y_train = y

        self.title = title
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

    def setup_ax(self, xlabel=None, ylabel=None):
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        if xlabel:
            ax.set_xlabel("Data dimension 1")
        if ylabel:
            ax.set_ylabel("Data dimension 2")
        if self.axis_off:
            ax.axis("off")
        return ax

    def get_mesh_input(self, x):
        x_shift = self.border * np.ptp(x[:, 0])
        y_shift = self.border * np.ptp(x[:, 1])
        x_min, x_max = x[:, 0].min() - x_shift, x[:, 0].max() + x_shift
        y_min, y_max = x[:, 1].min() - y_shift, x[:, 1].max() + y_shift
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, self.resolution),
                             np.linspace(y_min, y_max, self.resolution))
        mesh_input = np.c_[xx.ravel(), yy.ravel()]
        return mesh_input, xx, yy

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
                plt.show(block=True)

    def on_train_end(self, trainer, pl_module):
        plt.show()


class VisGLVQ2D(Vis2DAbstract):
    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        ax = self.setup_ax(xlabel="Data dimension 1",
                           ylabel="Data dimension 2")
        self.plot_data(ax, x_train, y_train)
        self.plot_protos(ax, protos, plabels)
        x = np.vstack((x_train, protos))
        mesh_input, xx, yy = self.get_mesh_input(x)
        _components = pl_module.proto_layer._components
        mesh_input = torch.Tensor(mesh_input).type_as(_components)
        y_pred = pl_module.predict(mesh_input)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)

        self.log_and_display(trainer, pl_module)


class VisSiameseGLVQ2D(Vis2DAbstract):
    def __init__(self, *args, map_protos=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_protos = map_protos

    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

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
            mesh_input, xx, yy = self.get_mesh_input(x)
        else:
            mesh_input, xx, yy = self.get_mesh_input(x_train)
        _components = pl_module.proto_layer._components
        mesh_input = torch.Tensor(mesh_input).type_as(_components)
        y_pred = pl_module.predict_latent(mesh_input,
                                          map_protos=self.map_protos)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)

        self.log_and_display(trainer, pl_module)


class VisCBC2D(Vis2DAbstract):
    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

        x_train, y_train = self.x_train, self.y_train
        protos = pl_module.components
        ax = self.setup_ax(xlabel="Data dimension 1",
                           ylabel="Data dimension 2")
        self.plot_data(ax, x_train, y_train)
        self.plot_protos(ax, protos, "w")
        x = np.vstack((x_train, protos))
        mesh_input, xx, yy = self.get_mesh_input(x)
        _components = pl_module.component_layer._components
        y_pred = pl_module.predict(
            torch.Tensor(mesh_input).type_as(_components))
        y_pred = y_pred.cpu().reshape(xx.shape)

        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)

        self.log_and_display(trainer, pl_module)


class VisNG2D(Vis2DAbstract):
    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

        x_train, y_train = self.x_train, self.y_train
        protos = pl_module.prototypes
        cmat = pl_module.topology_layer.cmat.cpu().numpy()

        ax = self.setup_ax(xlabel="Data dimension 1",
                           ylabel="Data dimension 2")
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

        self.log_and_display(trainer, pl_module)


class VisImgComp(Vis2DAbstract):
    def __init__(self,
                 *args,
                 random_data=0,
                 dataformats="CHW",
                 nrow=2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.random_data = random_data
        self.dataformats = dataformats
        self.nrow = nrow

    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

        if self.show:
            components = pl_module.components
            grid = torchvision.utils.make_grid(components, nrow=self.nrow)
            plt.imshow(grid.permute((1, 2, 0)).cpu(), cmap=self.cmap)

        self.log_and_display(trainer, pl_module)

    def add_to_tensorboard(self, trainer, pl_module):
        tb = pl_module.logger.experiment

        components = pl_module.components
        grid = torchvision.utils.make_grid(components, nrow=self.nrow)
        tb.add_image(
            tag="Components",
            img_tensor=grid,
            global_step=trainer.current_epoch,
            dataformats=self.dataformats,
        )

        if self.random_data:
            ind = np.random.choice(len(self.x_train),
                                   size=self.random_data,
                                   replace=False)
            data_img = self.x_train[ind]
            grid = torchvision.utils.make_grid(data_img, nrow=self.nrow)
            tb.add_image(tag="Data",
                         img_tensor=grid,
                         global_step=trainer.current_epoch,
                         dataformats=self.dataformats)
