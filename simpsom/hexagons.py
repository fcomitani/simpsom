"""
Hexagonal tiling library

F. Comitani @2017-2021 
"""

from math import sqrt, radians
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

def coor_to_hex(x,y):

    """Convert Cartesian coordinates to hexagonal tiling coordinates.

        Args:
            x (float): position along the x-axis of Cartesian coordinates.
            y (float): position along the y-axis of Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
                
    """


    newy = y*2/sqrt(3)*3/4
    newx = x
    
    if y%2: newx += 0.5
    
    return [newx,newy]    
    

def plot_hex(fig, centers, weights, color_ex=False):
    
    """Plot an hexagonal grid based on the nodes positions and color the tiles
       according to their weights.

        Args:
            fig (matplotlib figure object): the figure on which the hexagonal grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell 
                to be plotted in the Hexagonal tiling space.
            weights (list, float): array contaning informations on the weigths of each cell, 
                to be plotted as colors.
            color_ex (bool): if true, plot the example dataset with colors.
            
        Returns:
            ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
                
    """

    ax = fig.add_subplot(111, aspect='equal')

    xpoints = [x[0]  for x in centers]
    ypoints = [x[1]  for x in centers]
    patches = []

    cmap = plt.get_cmap('viridis')

    for x,y,w in zip(xpoints,ypoints,weights):
        
        facecolor = w if color_ex else cmap(w)

        hexagon = RegularPolygon((x,y), numVertices=6, radius=.95/sqrt(3), 
                            orientation=radians(0), 
                            facecolor=facecolor)
        patches.append(hexagon) 

    p = PatchCollection(patches,  match_original=True)

    setarray = None if color_ex else np.array(weights)
    p.set_array(setarray)

    ax.add_collection(p)
      
    ax.axis('off')
    ax.autoscale_view()
    
    return ax
