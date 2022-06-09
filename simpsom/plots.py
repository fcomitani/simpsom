"""
Plotting functions.

F Comitani, SG Riva, A Tangherloni 
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_map(centers, feature, polygons_class,
                    show=True, print_out=False, 
                    file_name="./som_plot.png",
                    **kwargs): 
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
                - fontsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
        """
            
        if "figsize" not in kwargs.keys():
            kwargs["figsize"] = (5, 5)
        if "title" not in kwargs.keys():
            kwargs["title"] = "SOM"
        if "cbar_label" not in kwargs.keys():
            kwargs["cbar_label"] = "Feature value"
        if "fontsize" not in kwargs.keys():
            kwargs["fontsize"] = 12
    
        fig  = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))

        ax = polygons_class.draw_map(fig, centers, feature)
        ax.set_title(kwargs["title"], size=kwargs["fontsize"]*1.15)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.0)
        cbar=plt.colorbar(ax.collections[0], cax=cax)
        cbar.set_label(kwargs["cbar_label"], size=kwargs["fontsize"])
        cbar.ax.tick_params(labelsize=kwargs["fontsize"]*.85)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        plt.sca(ax)
            
        if not file_name.endswith((".png",".jpg",".pdf")):
            file_name+=".png" 

        if print_out==True:
            plt.savefig(file_name, bbox_inches="tight", dpi=300)
        if show==True:
            plt.show()
        if show!=False and print_out!=False:
            plt.clf()

def line_plot(y_val, x_val=None, show=True, print_out=False, 
                    file_name="./line_plot.png",
                    **kwargs):
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
                - fontsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
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
    
        fig  = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))
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

        if not file_name.endswith((".png",".jpg",".pdf")):
            file_name+=".png" 

        if print_out==True:
            plt.savefig(file_name, bbox_inches="tight", dpi=300)
        if show==True:
            plt.show()
        if show!=False and print_out!=False:
            plt.clf()