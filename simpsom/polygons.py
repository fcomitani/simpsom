"""
Set of classes for custom tiling.

F Comitani, SG Riva, A Tangherloni
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


class Polygon():
    """ General class to define a custom polygonal tiling. """

    topology = None


    def get_topology(self):
        """ Get information on the set topology. """

        return self.topology

    @staticmethod
    def to_tiles(coor):
        """ Convert 2D cartesian coordinates to tiling coordinates.

        Args:
            coor (tuple[float,..]): the Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
        """

        return np.array(coor)

    @staticmethod
    def _tile(coor, color, edgecolor=None):
        """ Set the tile shape for plotting.

        Args:
            coor (tuple[float, float]): positon of the tile in the plot figure.
            color (tuple[float,...]): color tuple.
            edgecolor (tuple[float,...]): border color tuple.

        Returns:
            (matplotlib patch object): the tile to add to the plot.
        """

        return RegularPolygon(coor, 
                              numVertices=4, 
                              radius=.95/np.sqrt(2), 
                              orientation=np.radians(45), 
                              facecolor=color,
                              edgecolor=edgecolor)

    @classmethod
    def draw_map(cls, fig, centers, feature):
        """Draw a grid based on the selected tiling, nodes positions and color the tiles
        according to a given feature.

        Args:
            fig (matplotlib figure object): the figure on which the hexagonal grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell 
                to be plotted in the Hexagonal tiling space.
            feature (list, float): array contaning informations on the weigths of each cell, 
                to be plotted as colors.
            
        Returns:
            ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.         
        """

        ax = fig.add_subplot(111, aspect="equal")

        xpoints = [x[0]  for x in centers]
        ypoints = [x[1]  for x in centers]
        patches = []

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#ffffff", alpha=1.)
        edgecolor = None

        if np.isnan(feature).all():
            edgecolor = "#555555"

        for x,y,f in zip(xpoints, ypoints, feature):
            patches.append(cls._tile((x,y),
                           color=cmap(f),
                           edgecolor=edgecolor)
                          ) 

        pc = PatchCollection(patches,  match_original=True)
        pc.set_array(np.array(feature))
        ax.add_collection(pc)
          
        ax.axis("off")
        ax.autoscale_view()
        
        return ax

    @staticmethod
    def distance_pbc(nodes, net_shape, distance_func):
        """ Manage distances with PBC based on the tiling.

        Args:
            nodes (tuple[numpy array, numpy array]): the nodes 
                whose distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.

        Returns:
            (float): the distance adjusted by PBC.
        """
        
        return  np.min([distance_func(nodes[0],nodes[1]),
                     distance_func(nodes[0],nodes[1]+(net_shape[0],0)),
                     distance_func(nodes[0],nodes[1]-(net_shape[0],0)),
                     distance_func(nodes[0],nodes[1]+(0,net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(0,net_shape[1])),
                     distance_func(nodes[0],nodes[1]+(net_shape[0],net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(net_shape[0],net_shape[1])),
                     distance_func(nodes[0],nodes[1]+(-net_shape[0],net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(-net_shape[0],net_shape[1]))])


class Squares(Polygon):
    """ Class to define a square tiling. """

    topology = "square"


class Hexagons(Polygon):
    """ Class to define a hexagonal tiling. """

    topology = "hexagonal"

    @staticmethod
    def to_tiles(coor):
        """Convert 2D cartesian coordinates to tiling coordinates.

        Args:
            coor (tuple[float,..]): the Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
        """

        newy = coor[1]*2/np.sqrt(3)*3/4
        newx = coor[0]
        
        if coor[1]%2: newx += 0.5

        return np.array((newx, newy))
    
    @staticmethod
    def _tile(coor, color, edgecolor=None):
        """ Set the hexagonal tile for plotting.

        Args:
            coor (tuple[float, float]): positon of the tile in the plot figure.
            color (tuple[float,...]): color tuple.
            edgecolor (tuple[float,...]): border color tuple.

        Returns:
            (matplotlib patch object): the tile to add to the plot.
        """

        return RegularPolygon(coor, 
                              numVertices=6, 
                              radius=.95/np.sqrt(3), 
                              orientation=np.radians(0), 
                              facecolor=color,
                              edgecolor=edgecolor)

    @staticmethod
    def distance_pbc(nodes, net_shape, distance_func):
        """ Manage distances with PBC based on the tiling.

        Args:
            nodes (tuple[numpy array, numpy array]): the nodes 
                whose distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.

        Returns:
            (float): the distance adjusted by PBC.
        """

        offset = 0 if net_shape[1]%2 == 0 else 0.5
        net_shape = (net_shape[0], net_shape[1]*2/np.sqrt(3)*3/4)

        return  np.min([distance_func(nodes[0],nodes[1]),
                     distance_func(nodes[0],nodes[1]+(net_shape[0],0)),
                     distance_func(nodes[0],nodes[1]-(net_shape[0],0)),
                     distance_func(nodes[0],nodes[1]+(offset,net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(offset,net_shape[1])),
                     distance_func(nodes[0],nodes[1]+(net_shape[0]+offset,net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(net_shape[0]+offset,net_shape[1])),
                     distance_func(nodes[0],nodes[1]+(-net_shape[0]+offset,net_shape[1])),
                     distance_func(nodes[0],nodes[1]-(-net_shape[0]+offset,net_shape[1]))])
