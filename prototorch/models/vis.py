import os

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from prototorch.utils.celluloid import Camera
from prototorch.utils.colors import color_scheme
from prototorch.utils.utils import (gif_from_dir, make_directory,
                                    prettify_string)
from torch.utils.data import DataLoader, Dataset


class VisWeights(pl.Callback):
    """Abstract weight visualization callback."""
    def __init__(
        self,
        data=None,
        ignore_last_output_row=False,
        label_map=None,
        project_mesh=False,
        project_protos=False,
        voronoi=False,
        axis_off=True,
        cmap="viridis",
        show=True,
        display_logs=True,
        display_logs_settings={},
        pause_time=0.5,
        border=1,
        resolution=10,
        interval=False,
        save=False,
        snap=True,
        save_dir="./img",
        make_gif=False,
        make_mp4=False,
        verbose=True,
        dpi=500,
        fps=5,
        figsize=(11, 8.5),  # standard paper in inches
        prefix="",
        distance_layer_index=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.ignore_last_output_row = ignore_last_output_row
        self.label_map = label_map
        self.voronoi = voronoi
        self.axis_off = True
        self.project_mesh = project_mesh
        self.project_protos = project_protos
        self.cmap = cmap
        self.show = show
        self.display_logs = display_logs
        self.display_logs_settings = display_logs_settings
        self.pause_time = pause_time
        self.border = border
        self.resolution = resolution
        self.interval = interval
        self.save = save
        self.snap = snap
        self.save_dir = save_dir
        self.make_gif = make_gif
        self.make_mp4 = make_mp4
        self.verbose = verbose
        self.dpi = dpi
        self.fps = fps
        self.figsize = figsize
        self.prefix = prefix
        self.distance_layer_index = distance_layer_index
        self.title = "Weights Visualization"
        make_directory(self.save_dir)

    def _skip_epoch(self, epoch):
        if self.interval:
            if epoch % self.interval != 0:
                return True
        return False

    def _clean_and_setup_ax(self):
        ax = self.ax
        if not self.snap:
            ax.cla()
        ax.set_title(self.title)
        if self.axis_off:
            ax.axis("off")

    def _savefig(self, fignum, orientation="horizontal"):
        figname = f"{self.save_dir}/{self.prefix}{fignum:05d}.png"
        figsize = self.figsize
        if orientation == "vertical":
            figsize = figsize[::-1]
        elif orientation == "horizontal":
            pass
        else:
            pass
        self.fig.set_size_inches(figsize, forward=False)
        self.fig.savefig(figname, dpi=self.dpi)

    def _show_and_save(self, epoch):
        if self.show:
            plt.pause(self.pause_time)
        if self.save:
            self._savefig(epoch)
        if self.snap:
            self.camera.snap()

    def _display_logs(self, ax, epoch, logs):
        if self.display_logs:
            settings = dict(
                loc="lower right",
                # padding between the text and bounding box
                pad=0.5,
                # padding between the bounding box and the axes
                borderpad=1.0,
                # https://matplotlib.org/api/text_api.html#matplotlib.text.Text
                prop=dict(
                    fontfamily="monospace",
                    fontweight="medium",
                    fontsize=12,
                ),
            )

            # Override settings with self.display_logs_settings.
            settings = {**settings, **self.display_logs_settings}

            log_string = f"""Epoch: {epoch:04d},
            val_loss: {logs.get('val_loss', np.nan):.03f},
            val_acc: {logs.get('val_acc', np.nan):.03f},
            loss: {logs.get('loss', np.nan):.03f},
            acc: {logs.get('acc', np.nan):.03f}
            """
            log_string = prettify_string(log_string, end="")
            # https://matplotlib.org/api/offsetbox_api.html#matplotlib.offsetbox.AnchoredText
            anchored_text = AnchoredText(log_string, **settings)
            self.ax.add_artist(anchored_text)

    def on_train_start(self, trainer, pl_module, logs={}):
        self.fig = plt.figure(self.title)
        self.fig.set_size_inches(self.figsize, forward=False)
        self.ax = self.fig.add_subplot(111)
        self.camera = Camera(self.fig)

    def on_train_end(self, trainer, pl_module, logs={}):
        if self.make_gif:
            gif_from_dir(directory=self.save_dir,
                         prefix=self.prefix,
                         duration=1.0 / self.fps)
        if self.snap and self.make_mp4:
            animation = self.camera.animate()
            vid = os.path.join(self.save_dir, f"{self.prefix}animation.mp4")
            if self.verbose:
                print(f"Saving mp4 under {vid}.")
            animation.save(vid, fps=self.fps, dpi=self.dpi)


class VisPointProtos(VisWeights):
    """Visualization of prototypes.
    .. TODO::
        Still in Progress.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Point Prototypes Visualization"
        self.data_scatter_settings = {
            "marker": "o",
            "s": 30,
            "edgecolor": "k",
            "cmap": self.cmap,
        }
        self.protos_scatter_settings = {
            "marker": "D",
            "s": 50,
            "edgecolor": "k",
            "cmap": self.cmap,
        }

    def on_epoch_start(self, trainer, pl_module, logs={}):
        epoch = trainer.current_epoch
        if self._skip_epoch(epoch):
            return True

        self._clean_and_setup_ax()

        protos = pl_module.prototypes
        labels = pl_module.proto_layer.prototype_labels.detach().cpu().numpy()

        if self.project_protos:
            protos = self.model.projection(protos).numpy()

        color_map = color_scheme(n=len(set(labels)),
                                 cmap=self.cmap,
                                 zero_indexed=True)
        # TODO Get rid of the assumption y values in [0, num_of_classes]
        label_colors = [color_map[l] for l in labels]

        if self.data is not None:
            x, y = self.data
            # TODO Get rid of the assumption y values in [0, num_of_classes]
            y_colors = [color_map[l] for l in y]
            # x = self.model.projection(x)
            if not isinstance(x, np.ndarray):
                x = x.numpy()

            # Plot data points.
            self.ax.scatter(x[:, 0],
                            x[:, 1],
                            c=y_colors,
                            **self.data_scatter_settings)

            # Paint decision regions.
            if self.voronoi:
                border = self.border
                resolution = self.resolution
                x = np.vstack((x, protos))
                x_min, x_max = x[:, 0].min(), x[:, 0].max()
                y_min, y_max = x[:, 1].min(), x[:, 1].max()
                x_min, x_max = x_min - border, x_max + border
                y_min, y_max = y_min - border, y_max + border
                try:
                    xx, yy = np.meshgrid(
                        np.arange(x_min, x_max, (x_max - x_min) / resolution),
                        np.arange(y_min, y_max, (x_max - x_min) / resolution),
                    )
                except ValueError as ve:
                    print(ve)
                    raise ValueError(f"x_min: {x_min}, x_max: {x_max}. "
                                     f"x_min - x_max is {x_max - x_min}.")
                except MemoryError as me:
                    print(me)
                    raise ValueError("Too many points. "
                                     "Try reducing the resolution.")
                mesh_input = np.c_[xx.ravel(), yy.ravel()]

                # Predict mesh labels.
                if self.project_mesh:
                    mesh_input = self.model.projection(mesh_input)

                y_pred = pl_module.predict(torch.Tensor(mesh_input))
                y_pred = y_pred.reshape(xx.shape)

                # Plot voronoi regions.
                self.ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)

                self.ax.set_xlim(left=x_min + 0, right=x_max - 0)
                self.ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

        # Plot prototypes.
        self.ax.scatter(protos[:, 0],
                        protos[:, 1],
                        c=label_colors,
                        **self.protos_scatter_settings)

        # self._show_and_save(epoch)

    def on_epoch_end(self, trainer, pl_module, logs={}):
        epoch = trainer.current_epoch
        self._display_logs(self.ax, epoch, logs)
        self._show_and_save(epoch)


class Vis2DAbstract(pl.Callback):
    def __init__(self,
                 data,
                 title="Prototype Visualization",
                 cmap="viridis",
                 border=1,
                 resolution=50,
                 show_protos=True,
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
        self.show_protos = show_protos
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
        ax.axis("off")
        if xlabel:
            ax.set_xlabel("Data dimension 1")
        if ylabel:
            ax.set_ylabel("Data dimension 2")
        return ax

    def get_mesh_input(self, x):
        x_min, x_max = x[:, 0].min() - self.border, x[:, 0].max() + self.border
        y_min, y_max = x[:, 1].min() - self.border, x[:, 1].max() + self.border
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / self.resolution),
                             np.arange(y_min, y_max, 1 / self.resolution))
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
        y_pred = pl_module.predict(torch.Tensor(mesh_input))
        y_pred = y_pred.reshape(xx.shape)
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
        x_train = pl_module.backbone(torch.Tensor(x_train)).detach()
        if self.map_protos:
            protos = pl_module.backbone(torch.Tensor(protos)).detach()
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        if self.show_protos:
            self.plot_protos(ax, protos, plabels)
            x = np.vstack((x_train, protos))
            mesh_input, xx, yy = self.get_mesh_input(x)
        else:
            mesh_input, xx, yy = self.get_mesh_input(x_train)
        y_pred = pl_module.predict_latent(torch.Tensor(mesh_input))
        y_pred = y_pred.reshape(xx.shape)
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
        y_pred = pl_module.predict(torch.Tensor(mesh_input))
        y_pred = y_pred.reshape(xx.shape)

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
