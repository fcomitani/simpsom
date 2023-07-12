import sys
from types import ModuleType
from typing import Optional

import numpy as np
from loguru import logger


class Distance:
    """ Container class for distance functions. """

    def __init__(self, xp: ModuleType = None) -> None:
        """ Instantiate the Distance class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """

        self.xp = xp

    def _euclidean_squared_distance_part(self, x: np.ndarray, w: np.ndarray,
                                         w_flat_sq: Optional[np.ndarray] = None) -> float:
        """ Calculate the partial squared L2 distance.

        Args:
            x (array): first array.
            w (array): second array.

        Returns:
            (float): the partial L2 squared distance between two
                provided arrays
        """

        w_flat = w.reshape(-1, w.shape[2])
        if w_flat_sq is None:
            w_flat_sq = self.xp.power(w_flat, 2).sum(axis=1, keepdims=True)
        cross_term = self.xp.dot(x, w_flat.T)
        return -2 * cross_term + w_flat_sq.T

    def _euclidean_squared_distance(self, x: np.ndarray, w: np.ndarray,
                                    w_flat_sq: Optional[np.ndarray] = None) -> float:
        """Calculate the full squared L2 distance.

        Args:
            x (array): first array.
            w (array): second array.

        Returns:
            (float): the full L2 squared distance between two
                provided arrays
        """
        x_sq = self.xp.power(x, 2).sum(axis=1, keepdims=True)
        return self._euclidean_squared_distance_part(x, w, w_flat_sq) + x_sq

    def euclidean_distance(self, x: np.ndarray, w: np.ndarray, w_flat_sq: np.ndarray) -> float:
        """Calculate the L2 distance between two arrays.

        Args:
            x(array): first array.
            w(array): second array.

        Returns:
            (float): the euclidean distance between two
                provided arrays
        """
        return self.xp.nan_to_num(
            self.xp.sqrt(
                self._euclidean_squared_distance(x, w, w_flat_sq)
            )
        )

    def cosine_distance(self, x: np.ndarray, w: np.ndarray, w_flat_sq: np.ndarray) -> float:
        """Calculate the cosine distance between two arrays.

        Args:
            x(array): first array.
            w(array): second array.

        Returns:
            (float): the euclidean distance between two
                provided arrays
        """

        x_sq = self.xp.power(x, 2).sum(axis=1, keepdims=True)

        w_flat = w.reshape(-1, w.shape[2])

        similarity = self.xp.nan_to_num(self.xp.dot(
            x, w_flat.T) / self.xp.sqrt(x_sq * w_flat_sq.T))

        return 1 - similarity

    def manhattan_distance(self, x: np.ndarray, w: np.ndarray) -> float:
        """Calculate Manhattan distance between two arrays.

        Args:
            x(array): first array.
            w(array): second array.

        Returns:
            (float): the manhattan distance
                between two provided arrays.
        """

        if self.xp.__name__ == "cupy":

            _manhattan_distance_kernel = self.xp.ReductionKernel(
                "T x, T w",
                "T y",
                "abs(x-w)",
                "a+b",
                "y = a",
                "0",
                "l1norm")

            d = _manhattan_distance_kernel(
                x[:, self.xp.newaxis, self.xp.newaxis, :],
                w[self.xp.newaxis, :, :, :],
                axis=3
            )

        else:
            d = self.xp.linalg.norm(
                x[:, self.xp.newaxis, self.xp.newaxis, :] -
                w[self.xp.newaxis, :, :, :],
                ord=1,
                axis=3
            )

        return d.reshape(x.shape[0], w.shape[0] * w.shape[1])

    def batchpairdist(self, x: np.ndarray, w: np.ndarray, sq: np.ndarray, metric: str) -> np.ndarray:
        """ Calculates distances betweens points in batches. Two array-like objects
        must be provided, distances will be calculated between all points in the
        first array and all those in the second array.

        Args:
            a(array): first array.
            b(array): second array.
            metric(string): distance metric.
                Accepted metrics are euclidean, manhattan, and cosine(default "euclidean").
        Returns:
            d(array or list): the calculated distances.
        """

        if metric == "euclidean":
            return self.euclidean_distance(x, w, sq)

        elif metric == "cosine":
            return self.cosine_distance(x, w, sq)

        elif metric == "manhattan":
            return self.manhattan_distance(x, w)

        logger.error("Available metrics are: " +
                     "\"euclidean\", \"cosine\" and \"manhattan\"")
        sys.exit(1)

    def pairdist(self, a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
        """ Calculates distances betweens points. Two array-like objects
        must be provided, distances will be calculated between all points in the
        first array and all those in the second array.

        Args:
            a(array): first array.
            b(array): second array.
            metric(string): distance metric.
                Accepted metrics are euclidean, manhattan, and cosine(default "euclidean").

        Returns:
            d(array or list): the calculated distances.
        """

        if metric == "euclidean":
            squares_a = self.xp.sum(self.xp.power(a, 2), axis=1, keepdims=True)
            squares_b = self.xp.sum(self.xp.power(b, 2), axis=1, keepdims=True)
            return self.xp.sqrt(squares_a + squares_b.T - 2 * a.dot(b.T))

        elif metric == "cosine":
            return 1 - self.xp.dot(a / self.xp.linalg.norm(a, axis=1)[:, None],
                                   (b / self.xp.linalg.norm(b, axis=1)[:, None]).T)

        elif metric == "manhattan":
            def func(x, y): return self.xp.sum(self.xp.abs(x.T - y), axis=-1)
            return self.xp.stack([func(a[i], b) for i in range(a.shape[0])])

        logger.error("Available metrics are: " +
                     "\"euclidean\", \"cosine\" and \"manhattan\"")
        sys.exit(1)
