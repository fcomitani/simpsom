from types import ModuleType
from typing import Union, Callable, Tuple

import numpy as np
from loguru import logger


class Neighborhoods:
    """ Container class with functions to calculate neihgborhoods. """

    def __init__(self, xp: ModuleType = None) -> None:
        """ Instantiate the Neighborhoods class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """

        self.xp = xp

    def gaussian(self, c: np.ndarray, n: np.ndarray,
                 denominator: float) -> np.ndarray:
        """ Gaussian neighborhood function.

        Args:
            c (np.ndarray): center point.
            n (np.ndarray): matrix of nodes positions.
            denominator (float): the 2sigma**2 value.

        Returns
            (np.ndarray): a matrix of distances.  
        """

        return self.xp.exp(-self.xp.power(n - c, 2) / denominator)

    def mexican_hat(self, c: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Mexican hat neighborhood function.

        Args:
            c (np.ndarray): center point.
            n (np.ndarray): matrix of nodes positions.

        Returns
            (np.ndarray): a matrix of distances.  
        """

        return self.xp.power(n - c, 2)

    def bubble(self, c: np.ndarray, n: np.ndarray,
               threshold: float) -> np.ndarray:
        """ Bubble neighborhood function.

        Args:
            c (np.ndarray): center point.
            n (np.ndarray): matrix of nodes positions.
            threshold (float): the bubble threshold.
        Returns
            (np.ndarray): a matrix of distances.  
        """

        return self.xp.abs(n - c) < threshold

    def neighborhood_caller(self, center: Tuple[np.ndarray], sigma: float,
                            xx: np.ndarray, yy: np.ndarray,
                            neigh_func: str, pbc_func: Union[Callable, None] = None) -> np.ndarray:
        """Returns a neighborhood selection on any 2d topology.

        Args:
            center (Tuple[np.ndarray]): index of the center point along the xx yy grid.
            sigma (float): standard deviation/size coefficient.
            xx (array): x coordinates in the grid mesh.
            yy (array): y coordinates in the grid mesh.
            nigh_func (str): neighborhood specific distance function name
                (choose among 'gaussian', 'mexican_hat' or 'bubble')
            pbc_function (Callable): function to extend a distance
                function to account for pbc, as defined in polygons

        Returns:
            (array): the resulting neighborhood matrix.
        """

        d = 2 * sigma ** 2

        nx = xx[self.xp.newaxis, :, :]
        ny = yy[self.xp.newaxis, :, :]
        cx = xx.T[center][:, self.xp.newaxis, self.xp.newaxis]
        cy = yy.T[center][:, self.xp.newaxis, self.xp.newaxis]

        if neigh_func == 'gaussian':
            shape_fun = lambda x, y: self.gaussian(x, y, denominator=d)
        elif neigh_func == 'mexican_hat':
            shape_fun = self.mexican_hat
        elif neigh_func == 'bubble':
            shape_fun = lambda x, y: self.bubble(x, y, threshold=sigma)
        else:
            logger.error("{} neighborhood function not recognized.".format(neigh_func) +
                         "Choose among 'gaussian', 'mexican_hat' or 'bubble'.")
            raise ValueError

        if pbc_func is not None:
            px, py = pbc_func((cx, cy), (nx, ny), (nx.shape[2], nx.shape[1]), shape_fun, self.xp)
        else:
            px = shape_fun(cx, nx)
            py = shape_fun(cy, ny)

        if neigh_func == 'mexican_hat':
            p = px + py
            p = self.xp.exp(-p / d) * (1 - 2 / d * p)
        else:
            p = px * py

        return p.transpose((0, 2, 1))
