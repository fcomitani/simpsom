"""
Square tiling library

F Comitani, SG Riva, A Tangherloni
"""

from math import sqrt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot_square(fig, centers, weights, color_ex=False):
    
    """Plot a square grid based on the nodes positions and color the tiles
       according to their weights.

        Args:
            fig (matplotlib figure object): the figure on which the square grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell 
                to be plotted in the Square tiling space.
            weights (list, float): array contaning informations on the weigths of each cell, 
                to be plotted as colors.
            color_ex (bool): if true, plot the example dataset with colors.
            
        Returns:
            ax (matplotlib axis object): the axis on which the square grid has been plotted.
                
    """

    ax = fig.add_subplot(111, aspect='equal')

    xpoints = [x[0]  for x in centers]
    ypoints = [x[1]  for x in centers]
    patches = []

    cmap = plt.get_cmap('viridis')

    for x,y,w in zip(xpoints,ypoints,weights):
        
        facecolor = w if color_ex else cmap(w)

        squares = Rectangle((x,y),
                             width=.95,
                             height=.95,
                             facecolor=facecolor)
        patches.append(squares) 

    p = PatchCollection(patches,  match_original=True)

    setarray = None if color_ex else np.array(weights)
    p.set_array(setarray)

    ax.add_collection(p)
      
    ax.axis('off')
    ax.autoscale_view()
    
    return ax
