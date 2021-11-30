"""
Parallelizable functions interface for SimpSOM
F. Comitani     @2020-2021
"""

class Interface:

    """ Interface for parallelizable functions. """

    def __init__(self):

        self.gpu = False

        self.num = None
        
        self.PCA = None
        self.pairdist = None
        self.distance_metrics = None
        self.cluster = None

    def get_value(self, var):
        """ Returns value of given variable,
            Useful for transferring from GPU
            to CPU.
            
        Args:
            var (any): input variable.
        Returns:
            (any): value of the input variable.
        """

        return var

class InterfaceCPU(Interface):

    """ Interface for CPU functions. """

    def __init__(self):
        """ Load the required CPU libraries. """

        super().__init__()

        import numpy

        from sklearn.decomposition import PCA
        from sklearn.metrics import pairwise_distances as pairdist
        from sklearn import cluster
       
        self.gpu              = False

        self.num              = numpy

        self.PCA              = PCA
        self.pairdist         = pairdist
        self.cluster_algo     = cluster


class InterfaceGPU(Interface):

    """ Interface for GPU functions. """

    def __init__(self):
        """ Load the required CPU libraries. """

        super().__init__()

        import cupy

        from cuml.decomposition import PCA
        from cuml.metrics import pairwise_distances as pairdist
        from cuml import cluster

        self.gpu              = True

        self.num              = cupy

        self.PCA              = PCA
        self.pairdist         = pairdist
        self.cluster_algo     = cluster


    def get_value(self, var, pandas=False):
        """ Returns value of given variable,
            transferring it from GPU to CPU.

        Args:
            var (any): input variable.
            pandas (bool): if True, transform cudf to pandas.
        Returns:
            (any): value of the input variable.
        """

        if isinstance(var, self.num.ndarray):
            return self.num.asnumpy(var)

        return var.get()

if __name__ == "__main__":

    pass
