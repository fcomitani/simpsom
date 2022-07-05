"""
Neighborhood functions.

F Comitani, SG Riva, A Tangherloni 
"""

class Neighborhoods:
    """ Container class with functions to calculate neihgborhoods. """

    def __init__(self, xp=None):
        """ Instantiate the Neighborhoods class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """

        self.xp = xp

    #Why is this necessary?
    def prepare_neig_func(self, func, *first_args):
        """ Handles inputs for neighbourhood functions.

        Args:
            func (Callable): the neighborhood function.
            first_args (Iterable): the arguments to the
                neighborhood function.

        Returns:
            (array): the resulting neighborhood matrix.
        """

        def _inner(*args, **kwargs):
            """ Splits inputs for neighbourhood functions.
            
            Args:
                args (Iterable): the arguments to the
                    neighborhood function.
                kwargs (dict): additional keyword arguments.
            
            Returns:
                (array): the resulting neihborhood matrix.
            """

            return func(*first_args, *args, **kwargs)

        return _inner

    def gaussian(self, xx, yy, std_coeff, compact_support, c, sigma):
        """Returns a Gaussian centered in c on any 2d topology.
        
        TODO: remove compact_support and sigma
        TODO: what is sigma? sigma is supposed to be std_coeff

        Args:
            xx (array): x coordinates in the grid mesh.
            yy (array): y coordinates in the grid mesh.
            std_coeff (float): standard deviation coefficient.
            c (int): index of the center point along the xx yy grid.
            sigma (float): sigma coefficient.

        Returns:
            (array): the resulting neighborhood matrix.
        """

        d = 2*std_coeff**2*sigma**2

        nx = xx[self.xp.newaxis,:,:]
        ny = yy[self.xp.newaxis,:,:]
        cx = xx.T[c][:, self.xp.newaxis, self.xp.newaxis]
        cy = yy.T[c][:, self.xp.newaxis, self.xp.newaxis]

        px = self.xp.exp(-self.xp.power(nx-cx, 2)/d)
        py = self.xp.exp(-self.xp.power(ny-cy, 2)/d)

        if compact_support:
            px *= self.xp.logical_and(nx > cx-sigma, nx < cx+sigma)
            py *= self.xp.logical_and(ny > cy-sigma, ny < cy+sigma)

        return (px*py).transpose((0,2,1))


    def mexican_hat(self, xx, yy, std_coeff, compact_support, c, sigma):
        """Mexican hat centered in c on any topology.
        
        TODO: remove compact_support and sigma

        Args:
            xx (array): x coordinates in the grid mesh.
            yy (array): y coordinates in the grid mesh.
            std_coeff (float): standard deviation coefficient.
            c (int): index of the center point along the xx yy grid.
            sigma (float): sigma coefficient.

        Returns:
            (array): the resulting neighborhood matrix.
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
            py *= self.xp.logical_and(ny > cy-sigma, ny < cy+sigma)

        p = px + py
        
        return (self.xp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

    def bubble(self, neigx, neigy, c, sigma):
        """Constant function centered in c with spread sigma,
        which should be an odd value.
        
        TODO: remove compact_support

        Args:
            neigx (array): coordinates along the x axis.
            neigy (array): coordinates along the y axis.
            c (int): index of the center point along the xx yy grid.
            sigma (float): spread coefficient.

        Returns:
            (array): the resulting neighborhood matrix.
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