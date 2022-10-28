from types import ModuleType
from typing import Union, Tuple, Callable, Collection

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import RegularPolygon


class Polygon():
    """ General class to define a custom polygonal tiling. """

    topology = None

    def get_topology(self) -> None:
        """ Get information on the set topology. """

        return self.topology

    @staticmethod
    def to_tiles(coor: Tuple[float]) -> np.ndarray:
        """ Convert 2D cartesian coordinates to tiling coordinates.

        Args:
            coor (tuple[float,..]): the Cartesian coordinates.

        Returns:
            array: a 2d array containing the coordinates in the new space.
        """

        return np.array(coor)

    @staticmethod
    def _tile(coor: Tuple[float], color: Tuple[float],
              edgecolor: Tuple[float] = None) -> type(RegularPolygon):
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
                              radius=.95 / np.sqrt(2),
                              orientation=np.radians(45),
                              facecolor=color,
                              edgecolor=edgecolor)

    @classmethod
    def draw_map(cls, fig: Figure, centers: Collection[float],
                 feature: Collection[float], cmap: Union[ListedColormap, None] = None) -> plt.Axes:
        """Draw a grid based on the selected tiling, nodes positions and color the tiles
        according to a given feature.

        Args:
            fig (matplotlib figure object): the figure on which the hexagonal grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell
                to be plotted in the Hexagonal tiling space.
            feature (list, float): array contaning informations on the weigths of each cell,
                to be plotted as colors.
            cmap (ListedColormap): a custom color map.

        Returns:
            ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
        """

        ax = fig.add_subplot(111, aspect="equal")

        xpoints = [x[0] for x in centers]
        ypoints = [x[1] for x in centers]
        patches = []

        cmap = plt.get_cmap("viridis") if cmap is None else cmap
        cmap.set_bad(color="#ffffff", alpha=1.)
        edgecolor = None

        if np.isnan(feature).all():
            edgecolor = "#555555"

        for x, y, f in zip(xpoints, ypoints, feature):
            patches.append(cls._tile((x, y),
                                     color=cmap(f),
                                     edgecolor=edgecolor)
                           )

        pc = PatchCollection(patches, match_original=True, cmap=cmap)
        pc.set_array(np.array(feature))
        ax.add_collection(pc)

        ax.axis("off")
        ax.autoscale_view()

        return ax

    @staticmethod
    def distance_pbc(node_a: np.ndarray, node_b: np.ndarray, net_shape: Tuple[float],
                     distance_func: Callable, axis: Union[int, None] = None,
                     xp: ModuleType = np) -> float:
        """ Manage distances with PBC based on the tiling.

        Args:
            node_a (np.ndarray): the first node
                from which the distance will be calculated.
            node_b (np.ndarray): the second node
                from which the distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            axis (int): axis along which the minimum
                distance across PBC will be calculated.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (float): the distance adjusted by PBC.
        """

        net_shape = xp.array((net_shape[0], net_shape[1]))

        return xp.min(xp.array((distance_func(node_a, node_b),
                                distance_func(node_a, node_b +
                                              net_shape * xp.array((1, 0))),
                                distance_func(node_a, node_b -
                                              net_shape * xp.array((1, 0))),
                                distance_func(node_a, node_b +
                                              net_shape * xp.array((0, 1))),
                                distance_func(node_a, node_b -
                                              net_shape * xp.array((0, 1))),
                                distance_func(node_a, node_b + net_shape),
                                distance_func(node_a, node_b - net_shape),
                                distance_func(node_a, node_b +
                                              net_shape * xp.array((-1, 1))),
                                distance_func(node_a, node_b - net_shape * xp.array((-1, 1))))), axis=0)

    @staticmethod
    def neighborhood_pbc(center_node: Tuple[np.ndarray], nodes: Tuple[np.ndarray],
                         net_shape: Tuple[float], distance_func: Callable,
                         xp: ModuleType = np) -> np.ndarray:
        """ Manage neighborhood with PBC based on the tiling, adapted for
        batch training neighborhood functions. Works along a single
        provided axis and calculates the distance of a single node (center_node) from
        all other nodes in the network (nodes)

        Args:
            center_node (Tuple[np.ndarray]): position (index) of the first node along
                the provided axis. Shaped as (net_shape[1], 1, 1), for each axis.
            nodes (Tuple[np.ndarray]): the position of all nodes
                long a given axis as a matrix.
                Shaped as (1, net_shape[1], net_shape[0]), for each axis.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.

            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (np.ndarray): the distance from all nodes adjusted by PBC.
        """

        net_shape = (xp.full(nodes[0].shape, fill_value=net_shape[0]),
                     xp.full(nodes[1].shape, fill_value=net_shape[1]))

        return xp.max(xp.array((distance_func(center_node[0], nodes[0]),
                                distance_func(
                                    center_node[0], nodes[0] + net_shape[0]),
                                distance_func(center_node[0], nodes[0] - net_shape[0]))), axis=0), \
               xp.max(xp.array((distance_func(center_node[1], nodes[1]),
                                distance_func(
                                    center_node[1], nodes[1] + net_shape[1]),
                                distance_func(center_node[1], nodes[1] - net_shape[1]))), axis=0)


class Squares(Polygon):
    """ Class to define a square tiling. """

    topology = "square"


class Hexagons(Polygon):
    """ Class to define a hexagonal tiling. """

    topology = "hexagonal"

    @staticmethod
    def to_tiles(coor: Tuple[float]) -> np.ndarray:
        """Convert 2D cartesian coordinates to tiling coordinates.

        Args:
            coor (tuple[float,..]): the Cartesian coordinates.

        Returns:
            array: a 2d array containing the coordinates in the new space.
        """

        newy = coor[1] * 2 / np.sqrt(3) * 3 / 4
        newx = coor[0]

        if coor[1] % 2:
            newx += 0.5

        return np.array((newx, newy), dtype=np.float32)

    @staticmethod
    def _tile(coor: Tuple[float], color: Tuple[float],
              edgecolor: Tuple[float] = None) -> type(RegularPolygon):
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
                              radius=.95 / np.sqrt(3),
                              orientation=np.radians(0),
                              facecolor=color,
                              edgecolor=edgecolor)

    @staticmethod
    def distance_pbc(node_a: np.ndarray, node_b: np.ndarray, net_shape: Tuple[float],
                     distance_func: Callable, axis: Union[int, None] = None,
                     xp: ModuleType = np, ) -> float:
        """ Manage distances with PBC based on the tiling.

        Args:
            node_a (np.ndarray): the first node
                from which the distance will be calculated.
            node_b (np.ndarray): the second node
                from which the distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            axis (int): axis along which the minimum
                distance across PBC will be calculated.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (float): the distance adjusted by PBC.
        """

        offset = 0 if net_shape[1] % 2 == 0 else 0.5
        offset = xp.array((offset, 0))
        net_shape = xp.array((net_shape[0], net_shape[1] * 2 / np.sqrt(3) * 3 / 4))

        return xp.min(xp.array((distance_func(node_a, node_b),
                                distance_func(node_a, node_b +
                                              net_shape * xp.array((1, 0))),
                                distance_func(node_a, node_b -
                                              net_shape * xp.array((1, 0))),
                                distance_func(
                                    node_a, node_b + net_shape * xp.array((0, 1)) + offset),
                                distance_func(
                                    node_a, node_b - net_shape * xp.array((0, 1)) - offset),
                                distance_func(node_a, node_b + net_shape + offset),
                                distance_func(node_a, node_b - net_shape - offset),
                                distance_func(
                                    node_a, node_b + net_shape * xp.array((-1, 1)) + offset),
                                distance_func(node_a, node_b - net_shape * xp.array((-1, 1)) - offset))), axis=axis)

    @staticmethod
    def neighborhood_pbc(center_node: Tuple[np.ndarray], nodes: Tuple[np.ndarray],
                         net_shape: Tuple[float], distance_func: Callable,
                         xp: ModuleType = np) -> np.ndarray:
        """ Manage neighborhood with PBC based on the tiling, adapted for
        batch training neighborhood functions. Works along a single
        provided axis and calculates the distance of a single node (center_node) from
        all other nodes in the network (nodes)

        Args:
            center_node (Tuple[np.ndarray]): position (index) of the first node along
                the provided axis. Shaped as (net_shape[1], 1, 1), for each axis.
            nodes (Tuple[np.ndarray]): the position of all nodes
                long a given axis as a matrix.
                Shaped as (1, net_shape[1], net_shape[0]), for each axis.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (np.ndarray): the distance from all nodes adjusted by PBC.
        """

        offset = xp.full(nodes[0].shape, fill_value=0)
        if net_shape[1] % 2 != 0:
            offset[:] = 0.5

        net_shape = (xp.full(nodes[0].shape, fill_value=net_shape[0]),
                     xp.full(nodes[1].shape, fill_value=net_shape[1] * 2 / xp.sqrt(3) * 3 / 4))

        return xp.max(xp.array((distance_func(center_node[0], nodes[0]),
                                distance_func(
                                    center_node[0], nodes[0] + net_shape[0] + offset),
                                distance_func(center_node[0], nodes[0] - net_shape[0] - offset))), axis=0), \
               xp.max(xp.array((distance_func(center_node[1], nodes[1]),
                                distance_func(
                                    center_node[1], nodes[1] + net_shape[1]),
                                distance_func(center_node[1], nodes[1] - net_shape[1]))), axis=0)
