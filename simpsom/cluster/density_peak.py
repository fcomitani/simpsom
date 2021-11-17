"""
Density Peak Clustering

A Rodriguez, A Laio,
Clustering by fast search and find of density peaks
SCIENCE, 1492, vol 322 (2014) 

F. Comitani @2017 
"""

import sys
import numpy as np
from operator import attrgetter
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl

class Point:
    """ Class for the points to cluster. """

    def __init__(self, coordinates):

        """Initialise the point.

        Args:
            coordinates (np.array): Array containing the point coordinates in N dimensions.

        """
        
        self.coor = []
        for c in coordinates:
            self.coor.append(c)

        """ Initialise empty density (rho), higher-density distance (delta), list of distances, 
            the nearest neighbour of higher density, and the cluster."""

        self.rho    = 0
        self.delta  = sys.maxsize
        self.dists  = {}
        self.nneigh = None
        self.cl     = [0]
        self.core   = True


    def set_dist(self, coll):
    
        """Calculate the distances from all other points in a Collection. [Deprecated]

        Args:
            coll (Collection): Collection containing all the points of the dataset used to calculate the distances. 

        """

        warnings.warn('Setting individual distances is deprecated, use the Collection.set_dists() instead!', DeprecationWarning)

        for p2 in coll.points:
                if self != p2: self.dists[p2] = dist(self,p2)


    def set_rho(self, coll, type_func='step'):

        """Calculate the density of the single point for a given dataset. [Deprecated]

        Args:
            coll (Collection): Collection containing all the points of the dataset used to calculate the density. 
            type_func (str): step function type (step, gaussian kernel or logistic).

        """

        warnings.warn('Setting individual rhos is deprecated, use the Collection.set_rhos() instead!', DeprecationWarning)

        if self not in coll.points:
            print('WARNING: calculating the density for a point that was not found in the dataset, make sure to be consistent with your data!')
        
        for p2 in coll.points:
            if self != p2: 
                if type_func == 'step':
                    self.rho = self.rho+step(self,p2,self.refd)
                elif type_func == 'gaussian':
                    self.rho = self.rho+gaussian(self,p2,self.refd)
                elif type_func == 'logistic':
                    self.rho = self.rho+logistic(self,p2,self.refd)
                else:
                    """ Raise exception if metric other then euclidean is used. """
                    raise NotImplementedError('Only step, gaussian kernel or logistic functions are implemented')


    def set_delta(self, coll):

        """Calculate the distance of the point from higher density points and set the nearest neighbour. [Deprecated]

        Args:
            coll (Collection): Collection containing all the points of the dataset used to calculate the distance.

        """
        
        warnings.warn('Setting individual deltas is deprecated, use the Collection.set_deltas() instead!', DeprecationWarning)
        
        if self not in coll.points:
            print('WARNING: calculating the distance for a point that was not found in the dataset, make sure to be consistent with your data!')

        mind = sys.maxsize
        dist_high, dist_low = [], []
        
        for p2 in coll.points:
            if self != p2:
                d = dist(self,p2)
                if self.rho < p2.rho:
                    dist_high.append(d)
                    """ Choose nearest neighbour and handle empty list """
                    if d < np.min(dist_high or [999]): self.nneigh=p2
                else:
                    distsLow.append(d)
        
        """ If the point has maximal rho, then return max distance """

        self.delta = np.min(dist_high) if len(dist_high) > 0 \
                                       else np.max(dist_low) 


class Collection:

    """Class for a collection of point objects. """

    def __init__(self, coor_array, type_func='gaussian', percent=0.02, PBC=False, net_height=0, net_width=0):
        
        """ Generate a collection of point objects from an array containing their coordinates.
    
        Args:
            coor_array (np.array): Array containing the coordinates of the points to cluster.
            type_func (str): step function for calculating rho (step, gaussian kernel or logistic).
            percent (float): average percentage of neighbours.
            PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
            net_height (int, optional): Number of nodes along the first dimension, required for PBC.
            net_width (int, optional): Numer of nodes along the second dimension, required for PBC.

        """
    
        self.points = []
        for coors in coor_array:
            self.points.append(Point(coors))   

        index=int(np.round(len(self.points)*percent))

        self.PBC=PBC
        self.net_height = net_height
        self.net_width  = net_width

        self.all_dists = []
        self.set_dists()

        self.all_dists.sort()
        self.refd = self.all_dists[index]
                
        """ Make sure rhos are set before setting deltas """

        self.set_rhos(type_func)
        self.set_deltas()   

        self.clusters = {}
        

    def set_dists(self):
    
        """Calculate the distance matrix for all points. """

        for p1 in self.points:
            for p2 in self.points:
                if self.points.index(p1) < self.points.index(p2): 
                    d = dist(p1,p2, 'euclid', self.PBC, self.net_height, self.net_width)
                    self.all_dists.append(d) 
                    p1.dists[p2] = d 
                    p2.dists[p1] = d


    def set_rhos(self, type_func='step'):
    
        """Calculate the density for each point in the dataset. 

        Args:
            type_func (str): step function type (step, gaussian or logistic)

        """

        for p1 in self.points:
            for p2 in self.points:
                if self.points.index(p1) < self.points.index(p2): 
                    if type_func == 'step':
                        p1.rho = p1.rho+step(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                        p2.rho = p2.rho+step(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                    elif type_func == 'gaussian':
                        p1.rho = p1.rho+gaussian(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                        p2.rho = p2.rho+gaussian(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                    elif type_func == 'logistic':
                        p1.rho = p1.rho+sigmoid(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                        p2.rho = p2.rho+sigmoid(p1,p2,self.refd,PBC=self.PBC,net_height=self.net_height,net_width=self.net_width)
                    else:
                        raise NotImplementedError('Only step, gaussian kernel or logistic functions are implemented')


    def set_deltas(self):

        """Calculate the distance from higher density points for each point in the dataset. """

        for p1 in self.points:
            for p2 in self.points:
                if self.points.index(p1) < self.points.index(p2): 
                    d = p1.dists[p2]
                    if p1.rho < p2.rho and d < p1.delta: 
                        p1.delta  = d
                        p1.nneigh = p2
                    elif p1.rho > p2.rho and d < p2.delta: 
                        p2.delta  = d
                        p2.nneigh = p1
    
        """ If the point has maximal rho, then return max distance """

        pmax = max(self.points, key=attrgetter('rho'))
        pmax.delta = max(pmax.dists.values())


    def decision_graph(self, show=False, printout=True):

        """Calculate the decision graph, delta vs rho for the points belonging to the Collection 
            and find the cluster centers.

            Args:
                show (bool, optional): Choose to display the plot.
                printout (bool, optional): Choose to save the plot to a file.

        """

        fig, ax = plt.subplots()
        p_rhos, p_deltas = [p.rho for p in self.points], [p.delta for p in self.points]

        mean_deltas, stdev_deltas = np.mean(p_deltas), np.std(p_deltas)
        self.ctrs, ctrRhos, ctr_deltas = [], [], []
        
        i=1
        for p in self.points:
            if p.delta > mean_deltas+1.5*stdev_deltas:
                self.ctrs.append(p) 
                p.cl[0] = i
                i += 1
                ctrRhos.append(p.rho), ctr_deltas.append(p.delta) 
                p_rhos.pop(p_rhos.index(p.rho)), p_deltas.pop(p_deltas.index(p.delta))

        plt.scatter(p_rhos, p_deltas, \
            alpha=0.8, s=100, edgecolors='none', color='#3333AA')

        plt.scatter(ctrRhos, ctr_deltas, \
            alpha=0.8, s=100, edgecolors='none', color='#AA3333')

        if printout is True:
            plt.savefig('decision_graph.png', bbox_inches='tight', dpi=600)
        if show is True:
            plt.show()
        plt.clf()


    def cluster_assign(self):

        """Assign a cluster to each point according according to its nearest neighbour with higher density."""

        """ Temporary workaround until I can make mutable p.cl work, SLOW! """

        while [0] in [p.cl for p in self.points]:   
            for p in self.points: 
                if p.cl == [0]: p.cl = p.nneigh.cl


    def core_assign(self):
        
        """Assign points as belonging to the core or the halo of a cluster."""

        for c in range(1,len(self.ctrs)+1):
            self.clusters[c] = []

            border = []
            for p in self.points:
                if p.cl == [c]:
                    self.clusters[c].append(p)
                    for key,value in p.dists.items():
                        if key.cl != c and value < self.refd:
                            border.append(p)
                            continue

            if border != []:
                pmax    = max(border, key=attrgetter('rho'))             
                ref_rho = pmax.rho

                for p in self.clusters[c]:
                    if p.rho < ref_rho: 
                        p.core=False   


    def get_clusterList(self):

        """ Returns the indeces of the clustered points as a list.
            
            Returns:
                clusters (list, int): a list of lists containing the points indices belonging to each cluster
        """ 

        clusters = []

        for val in self.clusters.values():
            inds = []
            for p in val:
                inds.append(self.points.index(p))
            clusters.append(inds)

        return clusters 


def dist(p1,p2, metric='euclid', PBC=False, net_height=0, net_width=0):

    """Calculate the distance between two point objects in a N dimensional space according to a given metric.

    Args:
        p1 (point): First point object for the distance.
        p2 (point): Second point object for the distance.
        metric (string): Metric to use. For now only euclidean distance is implemented.
        PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
        net_height (int, optional): Number of nodes along the first dimension, required for PBC.
        net_width (int, optional): Numer of nodes along the second dimension, required for PBC.

    Returns:
        (float): The distance between the two points.

    """

    if metric == 'euclid':
        if len(p1.coor) != len(p2.coor): raise ValueError('Points must have the same dimensionality!')
        else:
            if PBC is True:
                """ Hexagonal Periodic Boundary Conditions """

                if net_height % 2 == 0:
                    offset = 0
                else: 
                    offset = 0.5
            
                return np.min([np.sqrt((p1.coor[0]-p2.coor[0])*(p1.coor[0]-p2.coor[0])\
                        +(p1.coor[1]-p2.coor[1])*(p1.coor[1]-p2.coor[1])),
                    #right
                    np.sqrt((p1.coor[0]-p2.coor[0]+net_width)*(p1.coor[0]-p2.coor[0]+net_width)\
                        +(p1.coor[1]-p2.coor[1])*(p1.coor[1]-p2.coor[1])),
                    #bottom 
                    np.sqrt((p1.coor[0]-p2.coor[0]+offset)*(p1.coor[0]-p2.coor[0]+offset)\
                        +(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)),
                    #left
                    np.sqrt((p1.coor[0]-p2.coor[0]-net_width)*(p1.coor[0]-p2.coor[0]-net_width)\
                        +(p1.coor[1]-p2.coor[1])*(p1.coor[1]-p2.coor[1])),
                    #top 
                    np.sqrt((p1.coor[0]-p2.coor[0]-offset)*(p1.coor[0]-p2.coor[0]-offset)\
                        +(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4)),
                    #bottom right
                    np.sqrt((p1.coor[0]-p2.coor[0]+net_width+offset)*(p1.coor[0]-p2.coor[0]+net_width+offset)\
                        +(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)),
                    #bottom left
                    np.sqrt((p1.coor[0]-p2.coor[0]-net_width+offset)*(p1.coor[0]-p2.coor[0]-net_width+offset)\
                        +(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]+net_height*2/np.sqrt(3)*3/4)),
                    #top right
                    np.sqrt((p1.coor[0]-p2.coor[0]+net_width-offset)*(p1.coor[0]-p2.coor[0]+net_width-offset)\
                        +(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4)),
                    #top left
                    np.sqrt((p1.coor[0]-p2.coor[0]-net_width-offset)*(p1.coor[0]-p2.coor[0]-net_width-offset)\
                        +(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4)*(p1.coor[1]-p2.coor[1]-net_height*2/np.sqrt(3)*3/4))])

            else:
                diffs = 0
                for i in range(len(p1.coor)): 
                    diffs = diffs+((p1.coor[i]-p2.coor[i])*(p1.coor[i]-p2.coor[i]))
                return np.sqrt(diffs)

    else:
        
        """ Raise exception if metric other then euclidean is used """
        
        raise NotImplementedError('PBC are implemented only with Euclidean distance')


def step(p1, p2, cutoff, PBC=False, net_height=0, net_width=0):

    """Step function activated when the distance of two points is less than the cutoff.

    Args:
        p1 (point): First point object for the distance.
        p2 (point): Second point object for the distance.
        cutoff (float): The cutoff to define the proximity of the points.
        PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
        net_height (int, optional): Number of nodes along the first dimension, required for PBC.
        net_width (int, optional): Numer of nodes along the second dimension, required for PBC.

    Returns:
        (int): 1 if the points are closer than the cutoff, 0 otherwise.

    """

    if dist(p1,p2, 'euclid', PBC, net_height, net_width)<cutoff: 
        return 1

    return 0  


def gaussian(p1, p2, sigma, PBC=False, net_height=0, net_width=0):

    """Gaussian function of the distance between two points scaled with sigma.

    Args:
        p1 (point): First point object for the distance.
        p2 (point): Second point object for the distance.
        sigma (float): The scaling factor for the distance.
        PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
        net_height (int, optional): Number of nodes along the first dimension, required for PBC.
        net_width (int, optional): Numer of nodes along the second dimension, required for PBC.

    Returns:
        (float): value of the gaussian function.

    """

    return np.exp(-1.0*dist(p1,p2, 'euclid', PBC, net_height, net_width)*\
            dist(p1,p2, 'euclid', PBC, net_height, net_width)/sigma*sigma)


def sigmoid(p1, p2, sigma, PBC=False, net_height=0, net_width=0):

    """Logistic function of the distance between two points scaled with sigma.

    Args:
        p1 (point): First point object for the distance.
        p2 (point): Second point object for the distance.
        sigma (float): The scaling factor for the distance.
        PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
        net_height (int, optional): Number of nodes along the first dimension, required for PBC.
        net_width (int, optional): Numer of nodes along the second dimension, required for PBC.

    Returns:
        (float): value of the logistic function.

    """

    return np.exp(-1.0*(1.0+np.exp((dist(p1,p2, 'euclid', PBC, net_height, net_width))/sigma)))


def density_peak(sample, show=False, printout=False, percent=0.02, PBC=False, net_height=0, net_width=0):

    """ Run the complete clustering algorithm in one go and returns the clustered indices as a list.

        Args:
            sample (array): The input dataset
            show (bool, optional): Choose to display the decision graph.
            printout (bool, optional): Choose to save the decision graph to a file.
        
        Returns:
            clusters (list, int): a list of lists containing the points indices belonging to each cluster
    """     
    
    pts = Collection(sample, percent=percent, PBC=PBC, net_height=net_height, net_width=net_width)
    pts.decision_graph(show=show, printout=printout)
    pts.cluster_assign()
    pts.core_assign()

    return pts.get_clusterList()


def dp_test(out_path='./'):

    import os

    """ Run the complete clustering algorithm on a test case and print the clustered points graph. 
    
        Args:
            out_path (str, optional): path to the output folder.
    """
    
    """ Set up output folder. """
    
    if out_path != './':
        try:
            os.makedirs(out_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    print("Testing Density Peak...")

    np.random.seed(100)
    samples1    = np.random.multivariate_normal([0, 0], [[1, 0.1],[0.1, 1]], 100)
    samples2    = np.random.multivariate_normal([10, 10], [[2, 0.5],[0.5, 2]], 100)
    samples3    = np.random.multivariate_normal([0, 10], [[2, 0.5],[0.5, 2]], 100)
    samples4    = np.random.uniform(0, 14, [50,2])
    samplesTmp  = np.concatenate((samples1,samples2), axis=0)
    samplesTmp2 = np.concatenate((samplesTmp,samples3), axis=0)
    samples     = np.concatenate((samplesTmp2,samples4), axis=0)

    pts = Collection(samples)

    pts.decision_graph(printout=False)
    pts.cluster_assign()
    pts.core_assign()

    print(pts.get_clusterList())

    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==0], [p.coor[1] for p in pts.points if p.cl[0]==0], 'o', c='black')
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==1 and p.core==True], [p.coor[1] for p in pts.points if p.cl[0]==1 and p.core==True], 'o', c="#ff0000")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==1 and p.core==False], [p.coor[1] for p in pts.points if p.cl[0]==1 and p.core==False], 'o', c="#ffaaaa")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==2 and p.core==True], [p.coor[1] for p in pts.points if p.cl[0]==2 and p.core==True], 'o', c="#00ff00")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==2 and p.core==False], [p.coor[1] for p in pts.points if p.cl[0]==2 and p.core==False], 'o', c="#aaffaa")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==3 and p.core==True], [p.coor[1] for p in pts.points if p.cl[0]==3 and p.core==True], 'o', c="#ffff00")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==3 and p.core==False], [p.coor[1] for p in pts.points if p.cl[0]==3 and p.core==False], 'o', c="#ffffaa")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==4 and p.core==True], [p.coor[1] for p in pts.points if p.cl[0]==4 and p.core==True], 'o', c="#0000ff")
    plt.plot([p.coor[0] for p in pts.points if p.cl[0]==4 and p.core==False], [p.coor[1] for p in pts.points if p.cl[0]==4 and p.core==False], 'o', c="#aaaaff")

    plt.savefig(os.path.join(out_path,'dp_test_out.png'), bbox_inches='tight', dpi=200)

    
    print("Done!")


if __name__ == "__main__":

    test()
