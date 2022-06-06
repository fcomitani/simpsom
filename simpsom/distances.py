import sys

def __euclidean_squared_distance_part(x, w, w_flat_sq=None, xp=None):
    """Calculate partial squared L2 distance
    This function does not sum x**2 to the result since it's not needed to 
    compute the best matching unit (it's not dependent on the neuron but
    it's a constant addition on the row).
    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = xp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = xp.dot(x, w_flat.T)
    return -2 * cross_term + w_flat_sq.T

def __euclidean_squared_distance(x, w, w_flat_sq=None, xp=None):
    """Calculate squared L2 distance
    NB: result shape is (N,X*Y)
    """
    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)
    return __euclidean_squared_distance_part(x, w, w_flat_sq, xp) + x_sq

def __euclidean_distance(x, w, w_flat_sq=None, xp=None):
    """Calculate L2 distance
    NB: result shape is (N,X*Y)
    """
    return xp.nan_to_num(
        xp.sqrt(
            __euclidean_squared_distance(x, w, w_flat_sq, xp)
        )
    )

def __cosine_distance(x, w, w_flat_sq=None, xp=None):
    """Calculate cosine distance
    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = xp.power(w_flat, 2).sum(axis=1, keepdims=True)

    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)

    num = xp.dot(x, w_flat.T)
    denum = xp.sqrt(x_sq * w_flat_sq.T)
    similarity = xp.nan_to_num(num/denum)

    return 1 - similarity

def __manhattan_distance(x, w, xp=None):
    """Calculate Manhattan distance
    It is very slow (~10x) compared to euclidean distance
    TODO: improve performance. Maybe a custom kernel is necessary
    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        
        _manhattan_distance_kernel = cp.ReductionKernel(
            'T x, T w',
            'T y',
            'abs(x-w)',
            'a+b',
            'y = a',
            '0',
            'l1norm')
        
        d = _manhattan_distance_kernel(
            x[:,xp.newaxis,xp.newaxis,:], 
            w[xp.newaxis,:,:,:], 
            axis=3
        )
        
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
    else:
        d = xp.linalg.norm(
            x[:,xp.newaxis,xp.newaxis,:]-w[xp.newaxis,:,:,:], 
            ord=1,
            axis=3,
        )
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
    
def batchpairdist(x, w, metric, sq=None, xp=None):

    if xp.__name__ not in ['numpy', 'cupy']:
        print("Error in importing numpy/cupy!")
        sys.exit()
    
    xp = __import__(xp.__name__)
        
    if metric not in ['cosine', 'euclidean', 'manhattan']:
        print("Chose a correct distance metric!")
        sys.exit()
    if metric=='cosine':
        return __cosine_distance(x, w, w_flat_sq=sq, xp=xp)
    elif metric=='euclidean':
        return __euclidean_distance(x, w, w_flat_sq=sq, xp=xp)
    else: # Slow
        return __manhattan_distance(x, w, xp=xp)
    
    
def pairdist(a, b, metric, cpu=False):

    """Calculating distances betweens points.

    Args:
        a (np or cp ..array): .
        b (np or cp ..array): .
        metric (string): distance metric. Accepted metrics are euclidean, manhattan, and cosine (default 'euclidean').

    Returns:
        d (np or cp ..array or list): distances. 

    """

    if metric not in ['cosine', 'euclidean', 'manhattan']:
        print("Chose a correct distance metric!")
        sys.exit()
    
    if cpu==False:
        try:
            import cupy as xp
        except:
            import numpy as xp
            cpu = True
    else:
        import numpy as xp

    if metric=='euclidean':
        squares_a = xp.sum(xp.power(a, 2), axis=1, keepdims=True)
        squares_b = xp.sum(xp.power(b, 2), axis=1, keepdims=True)
        d         = xp.sqrt(squares_a + squares_b.T - 2*a.dot(b.T))
    elif metric=='manhattan': # Slow
        funz = lambda x,y: xp.sum(xp.abs(x.T - y), axis=-1)
        d    = xp.stack([funz(a[i], b) for i in range(a.shape[0])])
    else:
        d = 1 - xp.dot(a/xp.linalg.norm(a,axis=1)[:,None],
                       (b/xp.linalg.norm(b,axis=1)[:,None]).T)

    return d