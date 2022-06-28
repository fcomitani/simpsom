"""
Neighborhood functions.
F Comitani, SG Riva, A Tangherloni 
"""

#ToDo: add docstring

class Neighborhoods:
    """ Kohonen SOM Network class. """

    def __init__(self, xp=None):
        self.xp = xp

    def prepare_neig_func(self, func, *first_args):
        def _inner(*args, **kwargs):
            return func(*first_args, *args, **kwargs)
        return _inner

    def gaussian(self, xx, yy, std_coeff, compact_support, c, sigma):
        """Returns a Gaussian centered in c on any topology
        
        TODO: this function is much slower than the _rect one
        """

        d = 2*std_coeff**2*sigma**2

        nx = xx[self.xp.newaxis,:,:]
        ny = yy[self.xp.newaxis,:,:]

        cx = xx.T[c][:, self.xp.newaxis, self.xp.newaxis]
        cy = yy.T[c][:, self.xp.newaxis, self.xp.newaxis]

        ax = self.xp.exp(-self.xp.power(nx-cx, 2)/d)
        ay = self.xp.exp(-self.xp.power(ny-cy, 2)/d)

        if compact_support:
            ax *= self.xp.logical_and(nx > cx-sigma, nx < cx+sigma)
            ay *= self.xp.logical_and(ny > cy-sigma, ny < cy+sigma)

        return (ax*ay).transpose((0,2,1))


    def mexican_hat(self, xx, yy, std_coeff, compact_support, c, sigma):
        """Mexican hat centered in c on any topology
        
        TODO: this function is much slower than the _rect one
        """

        d = 2*std_coeff**2*sigma**2

        nx = xx[self.xp.newaxis,:,:]
        ny = yy[self.xp.newaxis,:,:]
        cx = xx.T[c][:, self.xp.newaxis, self.xp.newaxis]
        cy = yy.T[c][:, self.xp.newaxis, self.xp.newaxis]

        px = self.xp.power(nx-cx, 2)
        py = self.xp.power(ny-cy, 2)

        if compact_support:
            px *= self.xp.logical_and(nx > cx-sigma, nx < cx+sigma)
            px *= self.xp.logical_and(ny > cy-sigma, ny < cy+sigma)
            
        p = px + py
        
        return (self.xp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

    def bubble(self, neigx, neigy, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        
        nx = neigx[self.xp.newaxis,:]
        ny = neigy[self.xp.newaxis,:]
        cx = c[0][:,self.xp.newaxis]
        cy = c[1][:,self.xp.newaxis]

        ax = self.xp.logical_and(nx > cx-sigma,
                            nx < cx+sigma)
        ay = self.xp.logical_and(ny > cy-sigma,
                            ny < cy+sigma)
        return (ax[:,:,self.xp.newaxis]*ay[:,self.xp.newaxis,:])