from typing import Union, Collection, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure

from pylettes import Distinct20

from simpsom.polygons import Polygon


def plot_map(centers: Collection[np.ndarray], feature: Collection[np.ndarray], polygons_class: Polygon,
             show: bool = True, print_out: bool = False,
             file_name: str = "./som_plot.png",
             **kwargs: Tuple[int]) -> Tuple[Figure,  plt.Axes]:
    """Plot a 2D SOM

    Args:
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        feature (list or array): The SOM node feature defining the color map
            (e.g. node.weights, node.diff)
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "SOM"
    if "cbar_label" not in kwargs.keys():
        kwargs["cbar_label"] = "Feature value"
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    fig = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))
    ax = polygons_class.draw_map(fig, centers, feature,
                                 cmap=kwargs['cmap'] if 'cmap' in kwargs 
                                                     else plt.get_cmap('viridis'))
    ax.set_title(kwargs["title"], size=kwargs["fontsize"]*1.15)

    divider = make_axes_locatable(ax)

    if not np.isnan(feature).all():
        cax = divider.append_axes("right", size="5%", pad=0.0)
        cbar = plt.colorbar(ax.collections[0], cax=cax)
        cbar.set_label(kwargs["cbar_label"], size=kwargs["fontsize"])
        cbar.ax.tick_params(labelsize=kwargs["fontsize"]*.85)
        cbar.outline.set_visible(False)

    fig.tight_layout()
    plt.sca(ax)

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax


def line_plot(y_val: Union[np.ndarray, list], x_val: Union[np.ndarray, list] = None,
              show: bool = True, print_out: bool = False,
              file_name: str = "./line_plot.png",
              **kwargs: Tuple[int]) -> Tuple[Figure,  plt.Axes]:
    """A simple line plot with maplotlib.

    Args: 
        y_val (array or list): values along the y axis.
        x_val (array or list): values along the x axis,
            if none, these will be inferred from the shape of y_val.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - xlabel (str): x-axis label,
            - ylabel (str): y-axis label,
            - logx (bool): if True set x-axis to logarithmic scale,
            - logy (bool): if True set y-axis to logarithmic scale,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "Line plot"
    if "xlabel" not in kwargs.keys():
        kwargs["xlabel"] = "x"
    if "ylabel" not in kwargs.keys():
        kwargs["ylabel"] = "y"
    if "logx" not in kwargs.keys():
        kwargs["logx"] = False
    if "logy" not in kwargs.keys():
        kwargs["logy"] = False
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    fig = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))
    ax = fig.add_subplot(111, aspect="equal")
    plt.sca(ax)
    plt.grid(False)

    if x_val is None:
        x_val = range(len(y_val))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(x_val, y_val, marker="o")

    plt.xticks(fontsize=kwargs["fontsize"]*.85)
    plt.yticks(fontsize=kwargs["fontsize"]*.85)

    if kwargs["logy"]:
        ax.set_yscale("log")

    if kwargs["logx"]:
        ax.set_xscale("log")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.xlabel(kwargs["xlabel"], fontsize=kwargs["fontsize"])
    plt.ylabel(kwargs["ylabel"], fontsize=kwargs["fontsize"])

    plt.title(kwargs["title"], size=kwargs["fontsize"]*1.15)

    ax.set_aspect("auto")
    fig.tight_layout()

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax


def scatter_on_map(datagroups: Collection[np.ndarray], centers: Collection[np.ndarray],
                   polygons_class: Polygon,
                   color_val: bool = None,
                   show: bool = True, print_out: bool = False,
                   file_name: str = "./som_scatter.png",
                   **kwargs: Tuple[int]) -> Tuple[Figure,  plt.Axes]:
    """Scatter plot with points projected onto a 2D SOM.

    Args:
        datagroups (list[array,...]): Coordinates of the projected points.
            This must be a nested list/array of arrays, where each element of 
            the list is a group that will be plotted separately.
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        color_val (array): The feature value to use as color map, if None
                the map will be plotted as white.
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "Projection onto SOM"
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    if color_val is None:
        color_val = np.full(len(centers), np.nan)

    fig, ax = plot_map(centers, color_val,
                       polygons_class,
                       show=False, print_out=False,
                       **kwargs)

    for i, group in enumerate(datagroups):
        ax.scatter(group[:, 0], group[:, 1],
                   color=Distinct20()[i % 20], edgecolor="#ffffff",
                   linewidth=1, label='{:d}'.format(i))

    plt.legend(bbox_to_anchor=(-.025, 1), fontsize=kwargs["fontsize"]*.85,
               frameon=False, title='Groups', ncol=int(len(datagroups)/10.0)+1,
               title_fontsize=kwargs["fontsize"])

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax
