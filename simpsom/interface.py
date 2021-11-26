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

        from cuml.decomposition import PCA as tSVD
        from cuml.metrics import pairwise_distances as pairdist
        from cuml import cluster

        self.gpu              = True

        self.num              = cupy

        self.PCA              = PCA
        self.pairdist         = pairdist
        self.cluster_algo     = cluster


if __name__ == "__main__":

    pass
