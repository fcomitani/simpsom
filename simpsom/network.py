"""
SimpSOM (Simple Self-Organizing Maps) v3.0.0
F Comitani, SG Riva, A Tangherloni
 
A lightweight python library for Kohonen Self-Organizing Maps (SOM).
"""

#logger
#unittest
#sqr and hex can be one function onlys
#adjust all headers
#adjust comments
#README
#example

from __future__ import print_function

import sys, time
import os, errno
import subprocess
import multiprocessing

import random 
import numpy as np
import pandas as pd
from math import sqrt, exp, log

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import simpsom.hexagons as hx
import simpsom.squares as sqr
import simpsom.distances as dist
import simpsom.neighborhoods as neighbor

import warnings
warnings.filterwarnings('ignore')

class SOMNet:
    """ Kohonen SOM Network class. """

    def __init__(self, net_height, net_width, data, load_file=None, metric='euclidean', topology="hexagonal", neighborhood_fun='gaussian',
                 init='random', PBC=False, GPU=False, CUML=False, random_seed=None, verbose=True):
        """Initialize the SOM network.

        Args:
            net_height (int): Number of nodes along the first dimension.
            net_width (int): Numer of nodes along the second dimension.
            data (array): N-dimensional dataset.
            load_file (str, optional): Name of file to load containing information 
                to initialize the network weights.
            metric (string): distance metric for the identification of best matching
                units. Accepted metrics are euclidean, manhattan, and cosine (default 'euclidean').
            topology (string): topology kohonen map. Accepted topology are hexagonal, and rectangular (default 'hexagonal').
            init (str or list of array): Nodes initialization method, to be chosen between 'random'
                or 'PCA' (default 'PCA'). Alternatively a couple of vectors can be provided
                whose values will be spanned uniformly.
            PBC (boolean): Activate/deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC (default 0).
            GPU (boolean): Activate/deactivate GPU run with RAPIDS (requires CUDA).
            random_seed (int): Seed for the random numbers generator (default None).   
        """
            
        self.verbose = bool(verbose)
        
        self.GPU  = bool(GPU)
        self.CUML = bool(CUML)

        if self.GPU:
            import cupy
            self.xp = cupy

            if self.CUML:
                try:
                    from cuml import cluster
                except:
                    print("WARNING: CUML libraries not found. Scikit-learn will be used instead.")
                    from sklearn import cluster

        else:
            from sklearn import cluster
            self.xp = np

        self.cluster_algo = cluster

        if random_seed is not None:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            self.xp.random.seed(random_seed)

        self.PBC = bool(PBC)
        if self.verbose:
            if self.PBC:
                print("Periodic Boundary Conditions active.")
            else:
                print("Periodic Boundary Conditions inactive.")

        self.node_list = []
        self.data = self.xp.array(data).astype(self.xp.float32)

        self.metric = metric
        self.topology = topology
        if self.topology=='hexagonal':
            self.polygon = hx 
        else:
            self.polygon = sqr

        self.neighborhood_fun = neighborhood_fun

        self.convergence = []

        self.net_height = net_height
        self.net_width  = net_width
        self._set_weights(load_file, init)

    def _set_weights(self, load_file, init):
        """Set initial map weights values, either by loading them from file or with random/PCA.

        Args:
            load_file (str, optional): Name of file to load containing information 
                to initialize the network weights.
            init (str or list of np or cp ..array): Nodes initialization method, to be chosen between 'random'
                or 'PCA' (default 'PCA'). Alternatively a couple of vectors can be provided
                whose values will be spanned uniformly.
        """

        init_vec = None
        init_bounds = None
        weights_array = None
        this_weight = None
        # When loaded from file, element 0 contains information on the network shape
        count_weight = 1

        if load_file is None:

            if init == 'PCA':
                if self.verbose:
                    print("The weights will be initialized with PCA.")
                if self.xp.__name__ == 'cupy':
                    init_vec = self.pca(self.data.get(), n_eigv=2)
                else:
                    init_vec = self.pca(self.data, n_eigv=2)
            
            elif init == 'random':
                if self.verbose:
                    print("The weights will be initialized randomly.")
                for i in range(self.data.shape[1]):
                    init_vec = [np.min(self.data, axis=0),
                                np.max(self.data, axis=0)]
            
            else:
                if self.verbose:
                    print("Custom weights provided.")
                init_vec = init

        else:   
            # TODO: add format checks

            if self.verbose:
                print("The weights will be loaded from file.\n"+ \
                    "The map shape will be overwritten and no weights"+ \
                    "initialization will be applied.")
            if not load_file.endswith('.npy'):
                load_file += '.npy' 
            weights_array = np.load(load_file)
            self.net_height = int(weights_array[0][0])
            self.net_width  = int(weights_array[0][1])
            self.PBC        = bool(weights_array[0][2])

        init_vec = self.xp.array(init_vec)

        for x in range(self.net_width):
            for y in range(self.net_height):

                if weights_array is not None:
                    this_weight = weights_array[count_wei]
                    count_weight += 1
 
                self.node_list.append(SOMNode(x, y, self.data.shape[1], 
                                              self.net_height, self.net_width, 
                                              self.PBC, self.topology,
                                              weight_bounds=init_bounds, 
                                              init_vec=init_vec, 
                                              weights_array=this_weight))

    @staticmethod
    def pca(A, n_eigv):
        
        """Generates PCA components to initialize network weights.

        Args:
            A (array): N-dimensional dataset.
            n_eigv (int): number of components to keep. 
                     
        Returns:            
            vectors (array): Principal axes in feature space, 
                representing the directions of maximum variance in the data.
        """
               
        M = np.mean(A.T, axis=1)
        C = A - M
        V = np.cov(C.T)

        return np.linalg.eig(V)[-1].T[:n_eigv]

    #ask simone
    def _get_n_process(self):
        
        if self.xp.__name__ == 'cupy':
            try:
                dev = self.xp.cuda.Device()
                n_smp = dev.attributes['MultiProcessorCount']
                max_thread_per_smp = dev.attributes['MaxThreadsPerMultiProcessor']
                return n_smp*max_thread_per_smp    
            except:
                print("There was an error in loading GPU processors information")
                return 0

        # not sure why any of this would be necessary
        # if cupy was loaded successfully get the n_process out
        # otherwise this shouldn't happen  
        #    try:
        #        return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
        #    except:
        #        print("Could not infer #cuda_cores")
        #        return 0
        else:
            try:
                #why 500
                return multiprocessing.cpu_count()*500
            except:
                print("Could not infer number of CPU_cores")
                return 0 
        
    #not sure about this, why not simply sample with replcement if 
    def _randomize_dataset(self, data, epochs):
        
        """Generates datapoints for online training.

        Args:
            data (array or list): N-dimensional dataset.
            epochs (int): Number of training iterations. 

        Returns:

        """
        
        if epochs < data.shape[0]:
            if self.verbose:
                print("Warning, epochs for online training are less than the entry datapoints!")
            
        dps = np.arange(0, data.shape[0], 1)
        if epochs <= data.shape[0]:
            entries = random.sample(dps.tolist(), k=epochs)
        else:
            entries = random.sample(dps.tolist(), k=data.shape[0])
            epcs = epochs - data.shape[0]
            while epcs > 0:
                entries += random.sample(dps.tolist(), k=data.shape[0])
                epcs -= data.shape[0]
        return entries
            
    # #unused function
    # def _find_label(self, c, i):
    #     for j in range(len(c)):
    #         if i in c[j]:
    #             return j            

    # #ask simone why dictionary and why is unused
    # def _sort_clusters(self, a, c):
    #     dict_labels = {}
    #     for i in range(a.shape[0]):
    #         label = self._find_label(c, i)
    #         dict_labels[i] = label
    #     return list(dict_labels.values())

    
    def plot_convergence(self, logax=False, print_out=False, out_path='./', 
                         fsize=(5, 5), figtitle=('Convergence', 14), fontsize=12):

        """ Plot the the map training progress according to the 
            chosen convergence criterion, when train_algo is batch.
            
        Args: 
            logax (bool, optional): if True, plot convergence on logarithmic y axis.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
            fsize (tuple(int, int), optional): Figure size.
            figtitle (tuple(str, str), optional): Figure title and fontsize.
            fontsize (int, optional): Legend fontsize.
        """
        
        if self.convergence==[]:
            if self.verbose:
                print("Warning, given the selected train_algo or the early_stop, there is no convergence values.")
            
        else:
            f, ax = plt.subplots(1,1,figsize=(fsize[0], fsize[1]))
            
            self.convergence = np.nan_to_num(self.convergence) 

            sns.lineplot(y=self.convergence, x=range(len(self.convergence)), marker="o", ax=ax)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            if logax:
                ax.set_yscale('log')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.xlabel('Iteration', fontsize=fontsize)
            plt.ylabel('Score', fontsize=fontsize)
            
            plt.title(figtitle[0], size=figtitle[1])
            
            plt.grid(False)
            plt.show()

            if print_out == True:
                print_name = os.path.join(out_path,'convergence.png')
                plt.savefig(print_name, bbox_inches='tight')
    
    
    def save(self, fileName='SOMNet_trained', out_path='./'):
 
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            fileName (str, optional): Name of file where the data will be saved.
            out_path (str, optional): Path to the folder where data will be saved.
        """
        
        weights_array = [self.xp.zeros(len(self.node_list[0].weights))]
        weights_array[0][0], weights_array[0][1], weights_array[0][2] = self.net_height, self.net_width, int(self.PBC)
        for node in self.node_list:
             weights_array.append(self.xp.asarray(node.weights))
        self.xp.save(os.path.join(out_path,fileName), self.xp.asarray(weights_array))
    

    def update_sigma(self, n_iter):
    
        """Update the gaussian sigma.

        Args:           
            n_iter (int): Iteration number.
            
        """
    
        self.sigma = self.start_sigma * exp(-n_iter/self.tau)
    

    def update_learning_rate(self, n_iter):
    
        """Update the learning rate.

        Args:           
            n_iter (int): Iteration number.
            
        """
        
        self.learning_rate =  self.start_learning_rate * exp(n_iter/self.epochs)
    
    
    def find_bmu_ix(self, vecs):
    
        """Find the index of the best matching unit (BMU) for a given list of vectors.

        Args:           
            vec (2d np or cp ..array or list of lists): vectors whose distance from the network
                nodes will be calculated.
            
        Returns:            
            bmu (SOMNode): The best matching unit node index.
            
        """


        dists = dist.pairdist(vecs,
                              self.xp.array([n.weights for n in self.node_list]), 
                              metric=self.metric, cpu=not(self.GPU))
        return self.xp.argmin(dists,axis=1)

    
    def train(self, train_algo='batch', epochs=-1, start_learning_rate=0.01, early_stop=None, 
              early_stop_patience=3, early_stop_tolerance=1e-4, batch_size=-1):
    
        """Train the SOM.

        Args:
            train_algo (str): training algorithm, choose between 'online' or 'batch' 
                (default 'online'). Beware that the online algorithm will run one datapoint
                per epoch, while the batch algorithm runs all points at one for each epoch.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints. 
            start_learning_rate (float): Initial learning rate, used only in online
                learning.
            early_stop (str): Early stopping method, for now only 'mapdiff' (checks if the
                weights of nodes don't change) and 'bmudiff' (checks if the assigned bmu to each sample
                don't change) are available. If None, don't use early stopping (default None).
            early_stop_patience (int): Number of iterations without improvement before stopping the 
                training, only available for batch training (default 3).
            early_stop_tolerance (float): Improvement tolerance, if the map does not improve beyond
                this threshold, the early stopping counter will be activated (it needs to be set
                appropriately depending on the used distance metric). Ignored if early stopping
                is off (default 1e-4).
            batch_size (int): Split the dataset in batches of this size when calculating the 
                new weights, works only when train_algo is 'batch' and helps keeping down the 
                memory requirements when working with large datasets, if -1 run the whole dataset
                at once. 

        """
        if self.verbose:
            print("The map will be trained with the "+train_algo+" algorithm.")
        self.start_sigma = max(self.net_height, self.net_width)/2
        self.start_learning_rate = start_learning_rate
        
        self.data = self.xp.array(self.data)
        
        if epochs == -1:
            epochs  = self.data.shape[0]*10
            
        self.epochs = epochs
        self.tau    = self.epochs/log(self.start_sigma)

        if batch_size == -1 or batch_size > self.data.shape[0]:
            _n_parallel = self._get_n_process()
        else:
            _n_parallel = batch_size
        
        if train_algo == 'online':
            """ Online training.
                Bootstrap: one datapoint is extracted randomly with replacement at each epoch 
                and used to update the weights.
            """
            
            datapoints = self._randomize_dataset(self.data, self.epochs)

            for n_iter in range(self.epochs):

                if n_iter%10==0:
                    if self.verbose:
                        print("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)), end=' ')

                self.update_sigma(n_iter)
                self.update_learning_rate(n_iter)
                
                datapoint = datapoints.pop()
                input_vec = self.data[datapoint, :].reshape(1,self.data.shape[1])
                
                bmu = self.node_list[int(self.find_bmu_ix(input_vec)[0])]

                for node in self.node_list:
                    node.update_weights(input_vec[0], self.sigma, self.learning_rate, bmu)

        elif train_algo == 'batch':
            """ Batch training.
                All datapoints are used at once for each epoch, 
                the weights are updated with the sum of contributions from all these points.
                No learning rate needed.

                Kinouchi, M. et al. "Quick Learning for Batch-Learning Self-Organizing Map" (2002).
            """

            #Storing the distances and weight matrices defeats the purpose of having
            #nodes as instances of a class, but it helps with the optimization
            #and parallelization at the cost of memory.
            #The object-oriented structure is kept to simplify code reading. 

            """ Calculate the square euclidean distance matrix to speed up batch training & store all weights as matrix. """

            all_weights = self.xp.array([n.weights for n in self.node_list], dtype=self.xp.float32)
            all_weights = all_weights.reshape(self.net_width, self.net_height, self.data.shape[1])
            early_stop_counter = 0

            numerator   = self.xp.zeros(all_weights.shape, dtype=self.xp.float32)
            denominator = self.xp.zeros((all_weights.shape[0], all_weights.shape[1], 1),dtype=self.xp.float32)

            unravel_precomputed = self.xp.unravel_index(self.xp.arange(self.net_width*self.net_height, dtype=self.xp.int64),
                                                        (self.net_width, self.net_height))
            _neigx = self.xp.arange(self.net_width)
            _neigy = self.xp.arange(self.net_height)
            _xx, _yy = self.xp.meshgrid(_neigx, _neigy)
            
            if self.neighborhood_fun == "bubble":
                neighborhood = neighbor.prepare_neig_func(neighbor.bubble, _neigx, _neigy, xp=self.xp)

            elif self.neighborhood_fun == "mexican":
                neighborhood = neighbor.prepare_neig_func(neighbor.mexican_hat, _xx, _yy, 0.5, False, xp=self.xp)

            else:
                neighborhood = neighbor.prepare_neig_func(neighbor.gaussian, _xx, _yy, 0.5, False, xp=self.xp)

            for n_iter in range(self.epochs):

                sq_weights = (self.xp.power(all_weights.reshape(-1, all_weights.shape[2]),2).sum(axis=1, keepdims=True))

                """ Early stop check. """

                if early_stop_counter == early_stop_patience:
                    
                    if self.verbose:
                        print("\rTolerance reached at epoch {:d}, stopping training.".format(n_iter-1))

                    break

                self.update_sigma(n_iter)
                self.update_learning_rate(n_iter)
                
                """ Matrix of gaussian effects. """

                if n_iter%10==0:
                    if self.verbose:
                        print("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)), end=' ')
                        
                """ Run through mini batches to ease the memory burden. """

                try: # reuse already allocated memory
                    numerator.fill(0)
                    denominator.fill(0)
                except AttributeError: # whoops, I haven't allocated it yet
                    numerator   = self.xp.zeros(all_weights.shape, dtype=self.xp.float32)
                    denominator = self.xp.zeros((all_weights.shape[0], all_weights.shape[1], 1),dtype=self.xp.float32)

                for i in range(0, len(self.data), _n_parallel):
                    start = i
                    end = start + _n_parallel
                    if end > len(self.data):
                        end = len(self.data)

                    batchdata = self.data[start:end]

                    """ Find BMUs for all points and subselect gaussian matrix. """
                    dists = dist.batchpairdist(batchdata, all_weights, self.metric, sq_weights, self.xp)

                    raveled_idxs = dists.argmin(axis=1)
                    wins = (unravel_precomputed[0][raveled_idxs], unravel_precomputed[1][raveled_idxs])

                    g_gpu = neighborhood(wins, self.sigma, xp=self.xp)*self.learning_rate
                    
                    sum_g_gpu = self.xp.sum(g_gpu, axis=0)
                    g_flat_gpu = g_gpu.reshape(g_gpu.shape[0], -1)
                    gT_dot_x_flat_gpu = self.xp.dot(g_flat_gpu.T, batchdata)

                    numerator   += gT_dot_x_flat_gpu.reshape(numerator.shape)
                    denominator += sum_g_gpu[:,:,self.xp.newaxis]
                    
            
                new_weights = self.xp.where(denominator != 0, numerator / denominator, all_weights)

                """ Early stopping, active if patience is not None """

                if early_stop is not None:

                    #These are pretty ruough convergence tests
                    #To do: add more
                    
                    if early_stop == 'mapdiff':      

                        """ Checks if the map weights are not moving. """

                        self.convergence.append(dist.pairdist(new_weights.reshape(self.net_width*self.net_height, self.data.shape[1]), 
                                                              all_weights.reshape(self.net_width*self.net_height, self.data.shape[1]), 
                                                              metric=self.metric).mean())
                    
                    elif early_stop == 'bmudiff':

                        """ Checks if the bmus mean distance from the samples has stopped improving. """

                        self.convergence.append(self.xp.min(dists, axis=1).mean())


                    else:
                        sys.exit('Error: convergence method not recognized. Choose between \'mapdiff\' and \'bmudiff\'.')

                    if n_iter > 0 and self.convergence[-2]-self.convergence[-1] < early_stop_tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                all_weights = new_weights

            """ Store final weights in the nodes objects. """
            #Revert back to object oriented
            
            all_weights = all_weights.reshape(self.net_width*self.net_height, self.data.shape[1])

            for n_iter, node in enumerate(self.node_list):
                node.weights = all_weights[n_iter] # * self.learning_rate

        else:
            sys.exit('Error: training algorithm not recognized. Choose between \'online\' and \'batch\'.')
        if self.verbose:               
            print("\rTraining SOM... done!")

        if self.GPU:
            for node in self.node_list:
                node.weights = node.weights.get()
        if early_stop is not None:
            if self.GPU:
                for n_iter, arr in enumerate(self.convergence):
                    
                    self.convergence[n_iter] = arr.get()
            
            
    def nodes_graph(self, colnum=0, show=False, print_out=False, out_path='./', colname=None, 
                    fsize=(5, 5), figtitle=('Node Grid w Feature', 14), 
                    cbartitle=('', 12), cbarticksize=10):
    
        """Plot a 2D map with hexagonal nodes and weights values

        Args:
            colnum (int): The index of the weight that will be shown as colormap.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.
            out_path (str, optional): Path to the folder where data will be saved.
            fsize (tuple(int, int), optional): Figure size.
            figtitle (tuple(str, str), optional): Figure title and fontsize.
            cbartitle (tuple(str, int), optional): cbar title and fontsize.
            cbarticksize (int, optional): cbar size.
        """
            
        if not colname:
            colname = str(colnum)
            
        if cbartitle[0] == '':
            cbartitle = (colname, cbartitle[1])
            
        centers = [[node.pos[0],node.pos[1]] for node in self.node_list]
        
        
        fig     = plt.figure(figsize=(fsize[0], fsize[1]))

        cols = [node.weights for node in self.node_list]

        cols = [c[colnum] for c in cols]

        ax = self.polygon.plot_map(fig, centers, cols)

        ax.set_title(figtitle[0], size=figtitle[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.0)
        cbar=plt.colorbar(ax.collections[0], cax=cax)
        cbar.set_label(cbartitle[0], size=cbartitle[1])
        cbar.ax.tick_params(labelsize=cbarticksize)
        cbar.outline.set_visible(False)

        plt.sca(ax)
        print_name=os.path.join(out_path,'nodes_feature_'+str(colnum)+'.png')
            
        if print_out==True:
            plt.savefig(print_name, bbox_inches='tight', dpi=dpi)
        if show==True:
            plt.show()
        if show!=False and print_out!=False:
            plt.clf()

            
    def difference_graph(self, show=False, print_out=False, returns=False, out_path='./', 
                    fsize=(5, 5), figtitle=('Node Grid w Feature', 14), 
                    cbartitle=('Weights Difference', 12), cbarticksize=10):
    
        """Plot a 2D map with nodes and weights difference among neighboring nodes.

        Args:
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            returns (bool, optional): Choose to return the difference value.
            out_path (str, optional): Path to the folder where data will be saved.
            fsize (tuple(int, int), optional): Figure size.
            figtitle (tuple(str, str), optional): Figure title and fontsize.
            cbartitle (tuple(str, int), optional): cbar title and fontsize.
            cbarticksize (int, optional): cbar size.
            
        Returns:
            (list): difference value for each node.             
        """
        
        """ Find adjacent nodes in the grid. """

        neighbors = [np.array([node2.weights for node2 in self.node_list
                                                   if node != node2 and node.get_node_distance(node2) <= 1.001])
                     for node in self.node_list]

        """ Calculate the summed weight difference. """

        diffs = [dist.pairdist(n.weights.reshape(1,n.weights.shape[0]), neighbors[i], metric='euclidean', cpu=True).mean()
                 for i, n in enumerate(self.node_list)]
        
        """ Define plotting hexagon centers. """

        centers = [[node.pos[0],node.pos[1]] for node in self.node_list]

        """ Set up and plot. """

        if show == True or print_out==True:
        
            fig     = plt.figure(figsize=(fsize[0], fsize[1]))

            ax = self.polygon.plot_map(fig, centers, diffs)

            ax.set_title(figtitle[0], size=figtitle[1])
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="5%", pad=0.0)
            cbar    = plt.colorbar(ax.collections[0], cax=cax)
            cbar.set_label(cbartitle[0], size=cbartitle[1])
            cbar.ax.tick_params(labelsize=cbarticksize)
            cbar.outline.set_visible(False)

            plt.sca(ax)
            print_name = os.path.join(out_path,'nodes_difference.png')
                        
            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight', dpi=dpi)
            if show == True:
                plt.show()
            if show != False and print_out != False:
                plt.clf()

        if returns == True:
            return diffs

        
    def project(self, array, colnum=-1, labels=[], show=False, print_out=False, returns=False, out_path='./', colname=None, 
                fsize=(5, 5), figtitle=('Datapoints Projection', 14), legendsize=10):

        """Project the datapoints of a given array to the 2D space of the 
            SOM by calculating the bmus. If requested plot a 2D map with as 
            implemented in nodes_graph and adds circles to the bmu
            of each datapoint in a given array.

        Args:
            array (np or cp ..array): An array containing datapoints to be mapped.
            colnum (int): The index of the weight that will be shown as colormap. 
                If not chosen, the difference map will be used instead.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
            colname (str, optional): Name of the column to be shown on the map.
            fsize (tuple(int, int), optional): Figure size.
            figtitle (tuple(str, str), optional): Figure title and fontsize.
            legendsize (int, optional): Legend fontsize.
            
        Returns:
            (list): bmu x,y position for each input array datapoint. 
            
        """
        
        if not colname:
            colname = str(colnum)
        
        if not isinstance(array, self.xp.ndarray):
            array = self.xp.array(array).astype(self.xp.float64)

        bmu_list, cls = [], []
        bmu_list = [self.node_list[int(mu)].pos for mu in self.find_bmu_ix(array)]

        if show == True or print_out == True:
        
            """ Call nodes_graph/difference_graph to first build the 2D map of the nodes. """
            
            df_plot = pd.DataFrame()
            
            f, ax = plt.subplots(1,1,figsize=(fsize[0], fsize[1]))

            #a random perturbation is added to the points positions so that data 
            #belonging plotted to the same bmu will be visible in the plot  
            df_plot['x']      = [pos[0]-0.125+random.random()*0.25 for pos in bmu_list]
            df_plot['y']      = [pos[1]-0.125+random.random()*0.25 for pos in bmu_list]
            df_plot['labels'] = labels

            self.difference_graph(False, False, False)
            sns.scatterplot(x='x', y='y', hue='labels', data=df_plot, palette="Paired", ax=ax)
            ax.set(xlabel=None)
            ax.set(ylabel=None)

            plt.title(figtitle[0], size=figtitle[1])    

            if colnum == -1:
                print_name = os.path.join(out_path,'projection_difference.png')
            else:   
                print_name = os.path.join(out_path,'projection_'+ colname +'.png')
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0, 
                      frameon=False, fontsize=legendsize, handles=handles[1:], labels=labels[1:]) 
               
            plt.ylim(0-0.5, self.net_height+0.5)
            plt.xlim(0-0.5, self.net_width+0.5)
            plt.grid(False)
            
            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight')
            if show == True:
                plt.show()
            # plt.clf()
        
        """ return x,y coordinates of bmus, useful for the clustering function. """
        
        if returns:
            return np.array(bmu_list)[:,:2]
        
        
    def cluster(self, array, clus_type='KMeans', num_cl=3,
                save_file=False, file_type='csv', show=True, print_out=False, out_path='./', returns=False, 
                fsize=(5, 5), figtitle=('Clusters', 14), legendsize=10):
    
        """Clusters the data in a given array according to the SOM trained map.
            The clusters can also be plotted.

        Args:
            array (np or cp ..array): An array containing datapoints to be clustered.
            clus_type (str, optional): The type of clustering to be applied, so far only quality threshold (qthresh) 
                algorithm is directly implemented, other algorithms require sklearn.
            num_cl (int, optional): The number of clusters for K-Means clustering
            save_file (bool, optional): Choose to save the resulting clusters in a text file.
            file_type (string, optional): Format of the file where the clusters will be saved (csv or dat)
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
            fsize (tuple(int, int), optional): Figure size.
            figtitle (tuple(str, str), optional): Figure title and fontsize.
            legendsize (int, optional): Legend fontsize.
            
        Returns:
            (list of int): A list containing the clusters of the input array datapoints.
            
        """

        """ Call project to first find the bmu for each array datapoint, but without producing any graph. """

        if array.shape[1] == 2:
            bmu_list = array
        else:
            bmu_list = self.project(array, returns=True)
        clusters = []

        if clus_type in ['KMeans', 'DBSCAN', 'AgglomerativeClustering']:
        
            """ Cluster according to algorithms implemented in sklearn, using defaul parameters. """
        
            if self.PBC == True:
                print("Warning: Only Quality Threshold and Density Peak clustering work with PBC")

            try:
                bmu_array = np.array(bmu_list)
                
                if clus_type == 'KMeans':
                    cl = self.cluster_algo.KMeans(n_clusters=num_cl).fit(bmu_array)
                    
                if clus_type == 'DBSCAN':
                    cl = self.cluster_algo.DBSCAN().fit(bmu_array)     
                
                if clus_type == 'AgglomerativeClustering':
                    cl = self.cluster_algo.AgglomerativeClustering(n_clusters=num_cl).fit(bmu_array)
                
                cl_labs = cl.labels_                 
                    
                for i in np.unique(cl_labs):
                    cl_list = []
                    tmp_list = range(len(bmu_list))
                    for j,k in zip(tmp_list,cl_labs):
                        if i == k:
                            cl_list.append(j)
                    clusters.append(cl_list)     
            except:
                print(('Unexpected error: ', sys.exc_info()[0]))
                raise
        else:
            sys.exit("Error: unkown clustering algorithm %s... Clustering algorithm must be 'KMeans', 'DBSCAN', or 'AgglomerativeClustering'"%clus_type)

        
        if save_file == True:
            with open(os.path.join(out_path,clus_type+'_clusters.'+file_type), 'w') as file:
                if file_type == 'csv':
                    separator = ','
                else: 
                    separator = '\t'
                for line in clusters:
                    for id in line: file.write(str(id)+separator)
                    file.write('\n')
        
        if print_out==True or show==True:

            print_name = os.path.join(out_path,clus_type+'_clusters.png')
            
            f, ax = plt.subplots(1,1,figsize=(fsize[0], fsize[1]))
            for ci, i in enumerate(range(len(clusters))):
                xc, yc  = [], []
                for c in clusters[i]:
                    #again, invert y and x to be consistent with the previous maps
                    xc.append(bmu_list[int(c)][0])
                    yc.append(self.net_height-bmu_list[int(c)][1])    
                sns.scatterplot(x=xc, y=yc, hue=i, palette=[sns.color_palette("Paired")[ci]], ax=ax)
                
            plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0, 
                       frameon=False, fontsize=legendsize) 

            plt.title(figtitle[0], size=figtitle[1])
            
            plt.ylim(0-0.5, self.net_height+0.5)
            plt.xlim(0-0.5, self.net_width+0.5)
            plt.grid(False)
            
            plt.gca().invert_yaxis()

            ax.set_yticklabels(np.arange(self.net_height+2, -2, -2))
            ax.set_xticklabels(np.arange(-2, self.net_width+2, 2))

            if show == True:
                plt.show()
            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight')
            # plt.clf()   
        
        if returns:
            return cl_labs

        
class SOMNode:

    """ Single Kohonen SOM Node class. """
    
    def __init__(self, x, y, num_weights, net_height, net_width, PBC, topology,
                weight_bounds=None, init_vec=None, weights_array=None,):
    
        """Initialize the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            num_weights (int): Length of the weights vector.
            net_height (int): Network height, needed for periodic boundary conditions (PBC)
            net_width (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            weight_bounds(np or cp ..array, optional): boundary values for the random initialization
                of the weights. Must be in the format [min_val, max_val]. 
                They are overwritten by 'init_vec'.
            init_vec (np or cp ..array, optional): Array containing the two custom vectors (e.g. PCA)
                for the weights initalization.
            weights_array (np or cp ..array, optional): Array containing the weights to give
                to the node if loaded from a file.
        """
        
        self.topology  = topology
        self.PBC       = PBC
        
        if self.topology == "hexagonal":
            self.pos = np.array(self.polygon.coor_to_hex(x,y))
        else:
            self.pos = np.array((x,y))

        self.weights   = []

        self.net_height = net_height
        self.net_width  = net_width

        if weights_array is not None:
            """ Load nodes's weights from file. """
            
            self.weights = weights_array

        elif init_vec is not None:
            """ Select uniformly in the space spanned by the custom vectors. """

            self.weights = ((x-self.net_width/2)*2.0/self.net_width*init_vec[0] + 
                            (y-self.net_height/2)*2.0/self.net_height*init_vec[1])

        elif weight_bounds is not None:
            """ Select randomly in the space spanned by the data. """
            
            for i in range(num_weights):
                self.weights.append(random.random()*(weight_bounds[1][i]-weight_bounds[0][i])+weight_bounds[0][i])
       
        else: 
            """ Else return error. """

            sys.exit(('Error in the network weights initialization, make sure to provide random initalization boundaries,\
                        custom vectors, or load the weights from file.'))
   
        # self.weights = np.array(self.weights)

    def get_node_distance(self, node):
    
        """Calculate the distance within the network between the node and another node.

        Args:
            node (SOMNode): The node from which the distance is calculated.
            
        Returns:
            (float): The distance between the two nodes.
            
        """

        x0 = self.pos[0]
        y0 = self.pos[1]
        x1 = node.pos[0]
        y1 = node.pos[1]

        if self.PBC:

            if self.topology == "hexagonal":
                
                offset = 0 if self.net_height % 2 == 0 else 0.5
                
                return  min([sqrt((x0-x1)**2+(y0-y1)**2),
                             #right
                             sqrt((x0-x1+self.net_width)**2+(y0-y1)**2),
                             #bottom 
                             sqrt((x0-x1+offset)**2+(y0-y1+self.net_height*2/sqrt(3)*3/4)**2),
                             #left
                             sqrt((x0-x1-self.net_width)**2+(y0-y1)**2),
                             #top 
                             sqrt((x0-x1-offset)**2+(y0-y1-self.net_height*2/sqrt(3)*3/4)**2),
                             #bottom right
                             sqrt((x0-x1+self.net_width+offset)**2+(y0-y1+self.net_height*2/sqrt(3)*3/4)**2),
                             #bottom left
                             sqrt((x0-x1-self.net_width+offset)**2+(y0-y1+self.net_height*2/sqrt(3)*3/4)**2),
                             #top right
                             sqrt((x0-x1+self.net_width-offset)**2+(y0-y1-self.net_height*2/sqrt(3)*3/4)**2),
                             #top left
                             sqrt((x0-x1-self.net_width-offset)**2+(y0-y1-self.net_height*2/sqrt(3)*3/4)**2)
                            ])

            elif self.topology == "rectangular":

                ### WRITE SQUARE HERE
                pass

            else:

                # This shouldn't happen
                print("WARNING: topology type not compatible with PBC.\n"+ \
                      "PBC will be turned off.")
                self.PBC = False
          
        else:
            return sqrt((x0-x1)**2+(y0-y1)**2)


    def update_weights(self, input_vec, sigma, learning_rate, bmu):
    
        """Update the node weights.

        Args:
            input_vec (np or cp ..array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            learning_rate (float): The updated learning rate.
            bmu (SOMNode): The best matching unit.
        """
    
        dist  = self.get_node_distance(bmu)
        gauss = exp(-dist**2/(2*sigma**2))
        
        self.weights -= gauss*learning_rate*(self.weights-input_vec)
        
        
if __name__ == "__main__":

    pass