"""
Distance functions.

F Comitani, SG Riva, A Tangherloni 
"""
#ToDo: add docstring.

import sys
from loguru import logger

class Distance:
    """ Container class for distance functions. """

    def __init__(self, xp=None):
        self.xp = xp

    def euclidean_distance(self, x, w, w_flat_sq=None):
        """Calculate L2 distance
        NB: result shape is (N,X*Y)
        """

        x_sq = self.xp.power(x, 2).sum(axis=1, keepdims=True)

        w_flat = w.reshape(-1, w.shape[2])
        if w_flat_sq is None:
            w_flat_sq = self.xp.power(w_flat, 2).sum(axis=1, keepdims=True)
        cross_term = self.xp.dot(x, w_flat.T)
        
        result = -2 * cross_term + w_flat_sq.T + x_sq

        return self.xp.nan_to_num(self.xp.sqrt(result))

    def cosine_distance(self, x, w, w_flat_sq=None):
        """Calculate cosine distance
        NB: result shape is (N,X*Y)
        """

        w_flat = w.reshape(-1, w.shape[2])
        if w_flat_sq is None:
            w_flat_sq = self.xp.power(w_flat, 2).sum(axis=1, keepdims=True)

        x_sq = self.xp.power(x, 2).sum(axis=1, keepdims=True)

        num = self.xp.dot(x, w_flat.T)
        denum = self.xp.sqrt(x_sq * w_flat_sq.T)
        similarity = self.xp.nan_to_num(num/denum)

        return 1 - similarity

    def manhattan_distance(self, x, w):
        """Calculate Manhattan distance
        It is very slow (~10x) compared to euclidean distance
        TODO: improve performance. Maybe a custom kernel is necessary
        NB: result shape is (N,X*Y)
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
                x[:,self.xp.newaxis,self.xp.newaxis,:], 
                w[self.xp.newaxis,:,:,:], 
                axis=3
            )
            
            return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
        else:
            d = self.xp.linalg.norm(
                x[:,self.xp.newaxis,self.xp.newaxis,:]-w[self.xp.newaxis,:,:,:], 
                ord=1,
                axis=3
            )
            return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
        
    def batchpairdist(self, x, w, metric, sq=None):
                
        if metric=="euclidean":
            return self.euclidean_distance(x, w, w_flat_sq=sq)
        elif metric=="cosine":
            return self.cosine_distance(x, w, w_flat_sq=sq)
        elif metric=="manhattan":
            return self.manhattan_distance(x, w)
        
        logger.error("Available metrics are: "+ \
                     "\"euclidean\", \"cosine\" and \"manhattan\"")
        sys.exit(1)

    def pairdist(self, a, b, metric):
        """Calculating distances betweens points.

        Args:
            a (array): .
            b (array): .
            metric (string): distance metric. 
                Accepted metrics are euclidean, manhattan, and cosine (default "euclidean").

        Returns:
            d (narray or list): distances. 

        """

        if metric=="euclidean":
            squares_a = self.xp.sum(self.xp.power(a, 2), axis=1, keepdims=True)
            squares_b = self.xp.sum(self.xp.power(b, 2), axis=1, keepdims=True)
            return self.xp.sqrt(squares_a + squares_b.T - 2*a.dot(b.T))      
        elif metric=="cosine":
            return 1 - self.xp.dot(a/self.xp.linalg.norm(a, axis=1)[:,None],
                           (b/self.xp.linalg.norm(b, axis=1)[:,None]).T)
        elif metric=="manhattan": 
            funz = lambda x,y: self.xp.sum(self.xp.abs(x.T - y), axis=-1)
            return self.xp.stack([funz(a[i], b) for i in range(a.shape[0])])
        
        logger.error("Available metrics are: "+ \
                     "\"euclidean\", \"cosine\" and \"manhattan\"")
        sys.exit(1)