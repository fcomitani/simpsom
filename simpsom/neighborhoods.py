"""
Neighborhood functions.

F Comitani, SG Riva, A Tangherloni 
"""

#ToDo: add docstring

def prepare_neig_func(func, *first_args, xp=None):
    def _inner(*args, **kwargs):
        return func(*first_args, *args, **kwargs)
    return _inner

def gaussian(xx, yy, std_coeff, compact_support, c, sigma, xp=None):
    """Returns a Gaussian centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """

    d = 2*std_coeff**2*sigma**2

    nx = xx[xp.newaxis,:,:]
    ny = yy[xp.newaxis,:,:]

    cx = xx.T[c][:, xp.newaxis, xp.newaxis]
    cy = yy.T[c][:, xp.newaxis, xp.newaxis]

    ax = xp.exp(-xp.power(nx-cx, 2, dtype=xp.float32)/d)
    ay = xp.exp(-xp.power(ny-cy, 2, dtype=xp.float32)/d)

    if compact_support:
        ax *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        ay *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return (ax*ay).transpose((0,2,1))


def mexican_hat(xx, yy, std_coeff, compact_support, c, sigma, xp=None):
    """Mexican hat centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """

    d = 2*std_coeff**2*sigma**2

    nx = xx[xp.newaxis,:,:]
    ny = yy[xp.newaxis,:,:]
    cx = xx.T[c][:, xp.newaxis, xp.newaxis]
    cy = yy.T[c][:, xp.newaxis, xp.newaxis]

    px = xp.power(nx-cx, 2, dtype=xp.float32)
    py = xp.power(ny-cy, 2, dtype=xp.float32)

    if compact_support:
        px *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        px *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)
        
    p = px + py
    
    return (xp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

def bubble(neigx, neigy, c, sigma, xp=None):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    
    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    ax = xp.logical_and(nx > cx-sigma,
                        nx < cx+sigma)
    ay = xp.logical_and(ny > cy-sigma,
                        ny < cy+sigma)
    return (ax[:,:,xp.newaxis]*ay[:,xp.newaxis,:]).astype(xp.float32)