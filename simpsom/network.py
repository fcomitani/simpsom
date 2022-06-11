"""
SimpSOM (Simple Self-Organizing Maps) v3.0.0 
A lightweight python library for Kohonen Self-Organizing Maps (SOM).

F Comitani, SG Riva, A Tangherloni
"""

# ToDo:
# - unittest
# - README
# - Docs: API + tutorial
# - PyPI
# - add PBC to clustering

from __future__ import print_function

import sys
import os 
import subprocess
import multiprocessing

from functools import partial

import random 
import numpy as np

from loguru import logger

import simpsom.distances as dist
import simpsom.neighborhoods as neighbor
from simpsom.polygons import Squares, Hexagons
from simpsom.plots import plot_map, line_plot, scatter_on_map

class SOMNet:
    """ Kohonen SOM Network class. """

    def __init__(self, net_height, net_width, data, load_file=None, metric="euclidean", topology="hexagonal", neighborhood_fun="gaussian",
                 init="random", PBC=False, GPU=False, CUML=False, random_seed=None, debug=False, output_path="./"):
        """Initialize the SOM network.

        Args:
            net_height (int): Number of nodes along the first dimension.
            net_width (int): Numer of nodes along the second dimension.
            data (array): N-dimensional dataset.
            load_file (str): Name of file to load containing information 
                to initialize the network weights.
            metric (string): distance metric for the identification of best matching
                units. Accepted metrics are euclidean, manhattan, and cosine (default "euclidean").
            topology (str): topology of the map tiling. 
                Accepted shapes are hexagonal, and square (default "hexagonal").
            neighborhood_fun (str): neighbours drop-off function for training, choose among gaussian,
                mexican_hat and bubble (default "gaussian").
            init (str or list[array, ...]): Nodes initialization method, choose between random
                or PCA (default "random"). Alternatively a couple of vectors can be provided
                whose values will be spanned uniformly.
            PBC (boolean): Activate/deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC (default False).
            GPU (boolean): Activate/deactivate GPU run with RAPIDS (requires CUDA, default False).
            CUML (boolean): Use CUML for clustering. If deactivate, use scikit-learn instead
                (requires CUDA, default False).
            random_seed (int): Seed for the random numbers generator (default None).   
            debug (bool): Set logging level printed to screen as debug.
            out_path (str): Path to the folder where all data and plots will be saved 
                (default, current folder).
        """
            
        self.output_path = output_path

        if not debug:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        logger.add(os.path.join(self.output_path,"som_{time}.log"), level="DEBUG")

        self.GPU  = bool(GPU)
        self.CUML = bool(CUML)

        if self.GPU:
            import cupy
            self.xp = cupy

            if self.CUML:
                try:
                    from cuml import cluster
                except:
                    logger.warning("CUML libraries not found. Scikit-learn will be used instead.")

        else:
            self.xp = np
            
        self.cluster_algo = cluster

        if random_seed is not None:
            os.environ["PYTHONHASHSEED"] = str(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            self.xp.random.seed(random_seed)

        self.PBC = bool(PBC)
        if self.PBC:
            logger.info("Periodic Boundary Conditions active.")

        self.node_list = []
        self.data = self.xp.array(data).astype(self.xp.float32)

        self.metric = metric

        if topology == "hexagonal":
            self.polygons = Hexagons
            logger.info("Hexagonal topology.")
        else:
            self.polygons = Squares
            logger.info("Square topology.")

        self.neighborhood_fun = neighborhood_fun

        self.convergence = []

        self.net_height = net_height
        self.net_width  = net_width
        self._set_weights(load_file, init)


    def _set_weights(self, load_file, init):
        """Set initial map weights values, either by loading them from file or with random/PCA.

        Args:
            load_file (str): Name of file to load containing information 
                to initialize the network weights.
            init (str or list[array, ..]): Nodes initialization method, to be chosen between "random"
                or "PCA" (default "PCA"). Alternatively a couple of vectors can be provided
                whose values will be spanned uniformly.
        """

        init_vec = None
        init_bounds = None
        weights_array = None
        this_weight = None

        # When loaded from file, element 0 contains information on the network shape
        count_weight = 1

        if load_file is None:

            if init == "PCA":
                logger.info("The weights will be initialized with PCA.")
                if self.xp.__name__ == "cupy":
                    init_vec = self.pca(self.data.get(), n_eigv=2)
                else:
                    init_vec = self.pca(self.data, n_eigv=2)
            
            elif init == "random":
                logger.info("The weights will be initialized randomly.")
                for i in range(self.data.shape[1]):
                    init_vec = [np.min(self.data, axis=0),
                                np.max(self.data, axis=0)]
            
            else:
                logger.info("Custom weights provided.")
                init_vec = init

        else:   
            # TODO: add format checks
            logger.info("The weights will be loaded from file.\n"+ \
                "The map shape will be overwritten and no weights"+ \
                "initialization will be applied.")
            if not load_file.endswith(".npy"):
                load_file += ".npy" 
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
                                              self.PBC, self.polygons,
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

    # ToDo: to be cleaned
    def _get_n_process(self):
        """ Count number of GPU or CPU processors. """

        if self.xp.__name__ == "cupy":
            try:
                dev = self.xp.cuda.Device() 
                n_smp = dev.attributes["MultiProcessorCount"]
                max_thread_per_smp = dev.attributes["MaxThreadsPerMultiProcessor"]
                return n_smp * max_thread_per_smp
            except:
                logger.error("Something went wrong when trying to count the number\n"+ \
                              "of GPU processors from CuPy.")
                return -1
                
            #the following block will never be executed, since both try and and except return something
            # try:
            #     return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
            # except:
            #     print("Could not infer #cuda_cores")
            #     return 0
        else:
            try:
                # Why * 500?
                return multiprocessing.cpu_count()*500
            except:
                logger.error("Something went wrong when trying to count the number\n"+ \
                              "of CPU processors.")
                return -1
                
        
    # not sure about the following, why not simply sample with replcement if 
    def _randomize_dataset(self, data, epochs):   
        """Generates a random list of datapoints indices for online training.

        Args:
            data (array or list): N-dimensional dataset.
            epochs (int): Number of training iterations. 

        Returns:
            entries (array): array with randomized indices
        """
        
        if epochs < data.shape[0]:
            logger.warning("Epochs for online training are less than the entry datapoints.")
            

        dps = np.arange(0, data.shape[0], 1)
        if epochs <= data.shape[0]:
            entries = random.sample(dps.tolist(), k=epochs)
        else:
            entries = random.sample(dps.tolist(), k=data.shape[0])
            epcs = epochs - data.shape[0]
            while epcs > 0:
                entries += random.sample(dps.tolist(), k=data.shape[0])
                epcs -= data.shape[0]
        
        # alternatively, if we don"t care running the whole dataset multiple times
        # we could just sample with replacement if epochs > data.shape[0]
        # replacement = False
        # if epochs > data.shape[0]:
        #     replacement = True
        # entries = random.sample(dps.tolist(), k=epochs, replacement=replacement)

        return entries 
    
    def save_map(self, file_name="./trained_som.npy"):
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            file_name (str): Name of the file where the data will be saved.
        """
        
        weights_array = [self.xp.zeros(len(self.node_list[0].weights))]
        weights_array[0][0], weights_array[0][1], weights_array[0][2] = self.net_height, self.net_width, int(self.PBC)
        for node in self.node_list:
             weights_array.append(node.weights)

        if not file_name.endswith((".npy")):
            file_name+=".npy"
        logger.info("Map shape and weights will be saved to:\n"+ \
                    os.path.join(self.output_path, file_name))
        np.save(os.path.join(self.output_path,file_name), np.asarray(weights_array))   

    def _update_sigma(self, n_iter):    
        """Update the gaussian sigma.

        Args:           
            n_iter (int): Iteration number.   
        """
    
        self.sigma = self.start_sigma * np.exp(-n_iter/self.tau)   

    def _update_learning_rate(self, n_iter):   
        """Update the learning rate.

        Args:           
            n_iter (int): Iteration number.   
        """
        
        self.learning_rate =  self.start_learning_rate * np.exp(n_iter/self.epochs)    
    
    def find_bmu_ix(self, vecs):
        """Find the index of the best matching unit (BMU) for a given list of vectors.

        Args:           
            vec (array or list[lists, ..]): vectors whose distance from the network
                nodes will be calculated.
            
        Returns:            
            bmu (SOMNode): The best matching unit node index.   
        """

        dists = dist.pairdist(vecs,
                              self.xp.array([n.weights for n in self.node_list]), 
                              metric=self.metric, xp=self.xp)
        return self.xp.argmin(dists,axis=1)

    
    def train(self, train_algo="batch", epochs=-1, start_learning_rate=0.01, early_stop=None, 
              early_stop_patience=3, early_stop_tolerance=1e-4, batch_size=-1):
        """Train the SOM.

        Args:
            train_algo (str): training algorithm, choose between "online" or "batch" 
                (default "online"). Beware that the online algorithm will run one datapoint
                per epoch, while the batch algorithm runs all points at one for each epoch.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints. 
            start_learning_rate (float): Initial learning rate, used only in online
                learning.
            early_stop (str): Early stopping method, for now only "mapdiff" (checks if the
                weights of nodes don"t change) and "bmudiff" (checks if the assigned bmu to each sample
                don"t change) are available. If None, don"t use early stopping (default None).
            early_stop_patience (int): Number of iterations without improvement before stopping the 
                training, only available for batch training (default 3).
            early_stop_tolerance (float): Improvement tolerance, if the map does not improve beyond
                this threshold, the early stopping counter will be activated (it needs to be set
                appropriately depending on the used distance metric). Ignored if early stopping
                is off (default 1e-4).
            batch_size (int): Split the dataset in batches of this size when calculating the 
                new weights, works only when train_algo is "batch" and helps keeping down the 
                memory requirements when working with large datasets, if -1 run the whole dataset
                at once. 
        """

        logger.info("The map will be trained with the "+train_algo+" algorithm.")
        self.start_sigma = max(self.net_height, self.net_width)/2
        self.start_learning_rate = start_learning_rate
        
        self.data = self.xp.array(self.data)
        
        if epochs == -1:
            epochs  = self.data.shape[0]*10
            
        self.epochs = epochs
        self.tau    = self.epochs/np.log(self.start_sigma)

        if batch_size == -1 or batch_size > self.data.shape[0]:
            _n_parallel = self._get_n_process()
        else:
            _n_parallel = batch_size
        
        if train_algo == "online":
            """ Online training.
                Bootstrap: one datapoint is extracted randomly with replacement at each epoch 
                and used to update the weights.
            """
            
            datapoints_ix = self._randomize_dataset(self.data, self.epochs)

            for n_iter in range(self.epochs):

                if n_iter%10==0:
                    logger.debug("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)))

                self._update_sigma(n_iter)
                self._update_learning_rate(n_iter)
                
                datapoint_ix = datapoints_ix.pop()
                input_vec = self.data[datapoint_ix, :].reshape(1,self.data.shape[1])
                
                bmu = self.node_list[int(self.find_bmu_ix(input_vec)[0])]

                for node in self.node_list:
                    node._update_weights(input_vec[0], self.sigma, self.learning_rate, bmu)

        elif train_algo == "batch":
            """ Batch training.
                All datapoints are used at once for each epoch, 
                the weights are updated with the sum of contributions from all these points.
                No learning rate needed.

                Kinouchi, M. et al. "Quick Learning for Batch-Learning Self-Organizing Map" (2002).
            """

            # WARNING: PBC currently not working for batch algo
            if self.PBC:
                logger.warning("PBC currently unavailable for batch training and will be turned off.")
                self.PBC = False

            # Storing the distances and weight matrices defeats the purpose of having
            # nodes as instances of a class, but it helps with the optimization
            # and parallelization at the cost of memory.
            # The object-oriented structure is kept to simplify code reading. 

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

                if early_stop_counter == early_stop_patience:
                    logger.info("\rEarly stop tolerance reached at epoch {:d}, stopping training.".format(n_iter-1))
                    break

                self._update_sigma(n_iter)
                self._update_learning_rate(n_iter)
                
                if n_iter%10==0:
                    logger.debug("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)))
                        
                # Run through mini batches to ease the memory burden.

                try: 
                    # Reuse already allocated memory
                    numerator.fill(0)
                    denominator.fill(0)
                except AttributeError: 
                    # I haven"t allocated it yet
                    numerator   = self.xp.zeros(all_weights.shape, dtype=self.xp.float32)
                    denominator = self.xp.zeros((all_weights.shape[0], all_weights.shape[1], 1),dtype=self.xp.float32)

                for i in range(0, len(self.data), _n_parallel):
                    start = i
                    end = start + _n_parallel
                    if end > len(self.data):
                        end = len(self.data)

                    batchdata = self.data[start:end]

                    # Find BMUs for all points and subselect gaussian matrix.
                    dists = dist.batchpairdist(batchdata, all_weights, self.metric, sq_weights, self.xp)

                    raveled_idxs = dists.argmin(axis=1)
                    wins = (unravel_precomputed[0][raveled_idxs], unravel_precomputed[1][raveled_idxs])

                    # ToDo: Add PBC here
                    g_gpu = neighborhood(wins, self.sigma, xp=self.xp)*self.learning_rate
                    
                    sum_g_gpu = self.xp.sum(g_gpu, axis=0)
                    g_flat_gpu = g_gpu.reshape(g_gpu.shape[0], -1)
                    gT_dot_x_flat_gpu = self.xp.dot(g_flat_gpu.T, batchdata)

                    numerator   += gT_dot_x_flat_gpu.reshape(numerator.shape)
                    denominator += sum_g_gpu[:,:,self.xp.newaxis]
                    
            
                new_weights = self.xp.where(denominator != 0, numerator / denominator, all_weights)

                if early_stop is not None:
                    # ToDo: These are pretty ruough convergence tests, add more
                    
                    if early_stop == "mapdiff":      
                        # Checks if the map weights are not moving. 
                        self.convergence.append(dist.pairdist(new_weights.reshape(self.net_width*self.net_height, self.data.shape[1]), 
                                                              all_weights.reshape(self.net_width*self.net_height, self.data.shape[1]), 
                                                              metric=self.metric, xp=self.xp).mean())
                    
                    elif early_stop == "bmudiff":
                        # Checks if the bmus mean distance from the samples has stopped improving. 
                        self.convergence.append(self.xp.min(dists, axis=1).mean())

                    else:
                        logger.error("Convergence method not recognized. Choose between \"mapdiff\" and \"bmudiff\".")
                        sys.exit(1)

                    if n_iter > 0 and self.convergence[-2]-self.convergence[-1] < early_stop_tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                all_weights = new_weights

            # Revert back to object oriented            
            all_weights = all_weights.reshape(self.net_width*self.net_height, self.data.shape[1])

            for n_iter, node in enumerate(self.node_list):
                node.weights = all_weights[n_iter] # * self.learning_rate

        else:
            logger.error("Training algorithm not recognized. Choose between \"online\" and \"batch\".")
            sys.exit(1)

        if self.GPU:
            for node in self.node_list:
                node.weights = node.weights.get()
        if early_stop is not None:
            if self.GPU:
                for n_iter, arr in enumerate(self.convergence):
                    self.convergence[n_iter] = arr.get()     

    def get_nodes_difference(self):
    
        """ Extracts the neighbouring nodes difference in weights and assigns it
        to each node object.
        """

        for node in self.node_list:
            neighbors = np.array([node2.weights for node2 in self.node_list
                                                   if node != node2 and node.get_node_distance(node2) <= 1.001])
            node._set_difference(dist.pairdist(node.weights.reshape(1, node.weights.shape[0]), neighbors, metric="euclidean", xp=self.xp).mean())
        logger.info('Weights difference among neighboring nodes calculated.')

    def project_onto_map(self, array, 
                file_name="./som_projected.npy"):

        """Project the datapoints of a given array to the 2D space of the 
        SOM by calculating the bmus.

        Args:
            array (array): An array containing datapoints to be mapped.
            file_name (str): Name of the file to which the data will be saved
                if not None.

        Returns:
            (list): bmu x,y position for each input array datapoint. 
        """

        if not isinstance(array, self.xp.ndarray):
            array = self.xp.array(array).astype(self.xp.float64)

        bmu_list, cls = [], []
        bmu_list = [self.node_list[int(mu)].pos for mu in self.find_bmu_ix(array)]

        if file_name is not None:
            if not file_name.endswith((".npy")):
                file_name+=".npy"
            logger.info("Projected coordinates will be saved to:\n"+ \
                        os.path.join(self.output_path, file_name))
            np.save(os.path.join(self.output_path, file_name), np.array(bmu_list))   

        return np.array(bmu_list, dtype=self.xp.float32)   

    def cluster(self, coor, project=True, clu_algo="DBSCAN", 
                file_name="./som_clusters.npy", **kwargs):
    
        """Project data onto the map and find clusters with scikit-learn clustering algorithms.
        ToDo: for the moment PBC will be ignored.

        Args:
            coor (array): An array containing datapoints to be mapped or
                pre-mapped if project False.
            project (bool): if True, project the points in coor onto the map.
            clu_algo (clustering obj or str): The clusters identification algorithm. A scikit-like
                class can be provided (must have a fit method), or a string among a pre-set list
                of algorithms ('KMeans', 'DBSCAN', 'AgglomerativeClustering')
            file_name (str): Name of the file to which the data will be saved
                if not None.
            kwargs (dict): Keyword arguments to the clustering algorithm:

        Returns:
            (list of int): A list containing the clusters of the input array datapoints.
            
        """

        bmu_coor = self.project_onto_map(coor) if project else coor

        if self.PBC:
            # ToDo: implement PBC, but basically providing the PBC-adjusted distance metric to 
            # the clustering algorithms
            logger.warning("PBC are not implemented with clustering yet and will be ignored for now.")
  
        if clu_algo == "KMeans":
            if 'n_clusters' not in kwargs.keys():
                kwargs['n_clusters'] = 3
            clu_algo = self.cluster_algo.KMeans(**kwargs)    
        elif clu_algo == "DBSCAN":
            clu_algo  = self.cluster_algo.DBSCAN(**kwargs)  
        elif clu_algo == "AgglomerativeClustering":
            if 'n_clusters' not in kwargs.keys():
                kwargs['n_clusters'] = 3
            clu_algo  = self.cluster_algo.AgglomerativeClustering(**kwargs)
        else:
            clu_algo = clu_algo(**kwargs)
            if not callable(getattr(self, "fit", None)):
                logger.error("ERROR: There was a problem with the clustering, make sure to provide a scikit-like clustering\n"+ \
                    "class or use one of among the preset list 'KMeans', 'DBSCAN', or 'AgglomerativeClustering',\n"+ \
                    "Custom classes must have a 'fit' method.")
                return None

        try:
            clu_labs = clu_algo.fit(bmu_coor).labels_
        except:
            logger.error("ERROR: There was a problem with the clustering, make sure to provide a scikit-like clustering\n"+ \
                    "class or use one of among the preset list 'KMeans', 'DBSCAN', or 'AgglomerativeClustering',\n"+ \
                    "Custom classes must have a 'fit' method.")
            return None
    
        if file_name is not None:
            if not file_name.endswith((".npy")):
                file_name+=".npy"
            logger.info("Clustering results will be saved to:\n"+ \
                        os.path.join(self.output_path, file_name))
            np.save(os.path.join(self.output_path, file_name), np.array(clu_labs))   

        return np.array(clu_labs), bmu_coor

    def plot_map_by_feature(self, feature, show=False, print_out=True,
                             **kwargs):
        """ Wrapper function to plot a trained 2D SOM map 
            color-coded according to a given feature.

        Args:
            feature (int): The feature number to use as color map.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot:
                - figsize (tuple(int, int)): the figure size,
                - title (str): figure title,
                - cbar_label (str): colorbar label,
                - labelsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
        """

        _, _ = plot_map([[node.pos[0],node.pos[1]] for node in self.node_list], 
                [node.weights[feature] for node in self.node_list], 
                self.polygons,
                show=show, print_out=print_out, 
                file_name=os.path.join(self.output_path, "./som_feature_{}.png".format(str(feature))),
                **kwargs) 

        if print_out:
            logger.info("Feature map will be saved to:\n"+ \
                        os.path.join(self.output_path, "./som_feature_{}.png".format(str(feature))))

    def plot_map_by_difference(self, show=False, print_out=True,
                             **kwargs):
        """ Wrapper function to plot a trained 2D SOM map 
            color-coded according neighbours weights difference.
            It will automatically calculate the difference values
            if not already computed.

        Args:
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot:
                - figsize (tuple(int, int)): the figure size,
                - title (str): figure title,
                - cbar_label (str): colorbar label,
                - labelsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
        """
        
        if self.node_list[0].difference is None:
            self.get_nodes_difference()

        if "cbar_label" not in kwargs.keys():
            kwargs["cbar_label"] = "Nodes difference value"

        _, _ = plot_map([[node.pos[0],node.pos[1]] for node in self.node_list], 
                [node.difference for node in self.node_list], 
                self.polygons,
                show=show, print_out=print_out, 
                file_name=os.path.join(self.output_path, "./som_difference.png"),
                **kwargs) 

        if print_out:
            logger.info("Node difference map will be saved to:\n"+ \
                        os.path.join(self.output_path, "./som_difference.png"))

    def plot_convergence(self, show=False, print_out=True,
                             **kwargs):

        """ Plot the the map training progress according to the 
            chosen convergence criterion, when train_algo is batch.
            
        Args:
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot:
                - figsize (tuple(int, int)): the figure size,
                - title (str): figure title,
                - xlabel (str): x-axis label,
                - ylabel (str): y-axis label,
                - logx (bool): if True set x-axis to logarithmic scale,
                - logy (bool): if True set y-axis to logarithmic scale,
                - fontsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
        """
        
        if len(self.convergence)==0:
            logger.warning("The current parameters yelded no convergence. The plot will not be produced.")
            
        else:

            conv_values = self.xp.nan_to_num(self.convergence) 

            if "title" not in kwargs.keys():
                kwargs["title"] = "Convergence"
            if "xlabel" not in kwargs.keys():
                kwargs["xlabel"] = "Iteration"
            if "ylabel" not in kwargs.keys():
                kwargs["ylabel"] = "Score"

            _, _ = line_plot(conv_values, 
                    show=show, print_out=print_out, 
                    file_name=os.path.join(self.output_path, "./som_convergence.png"),
                    **kwargs)

        if print_out:
            logger.info("Convergence results will be saved to:\n"+ \
                        os.path.join(self.output_path, "./som_convergence.png"))

    def plot_projected_points(self, coor, color_val=None,
                             project=True, jitter=True, 
                             show=False, print_out=True,
                             **kwargs):

        """Project points onto the trained 2D map and plot the result.

        Args:
            coor (array): An array containing datapoints to be mapped or
                pre-mapped if project False.
            color_val (array): The feature value to use as color map, if None
                the map will be plotted as white.
            project (bool): if True, project the points in coor onto the map.
            jitter (bool): if True, add jitter to points coordinates to help
                with overlapping points.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot:
                - figsize (tuple(int, int)): the figure size,
                - title (str): figure title,
                - cbar_label (str): colorbar label,
                - labelsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.
        """
        
        bmu_coor = self.project_onto_map(coor) if project else coor

        if jitter:
            bmu_coor += np.random.uniform(low=-.15, high=.15, size=(bmu_coor.shape[0],2))

        _, _ = scatter_on_map([bmu_coor], 
                       [[node.pos[0],node.pos[1]] for node in self.node_list],
                       self.polygons,
                       color_val=color_val, 
                       show=show, print_out=print_out,
                       file_name=os.path.join(self.output_path, "./som_projected.png"),
                       **kwargs)

        if print_out:
            logger.info("Projected data scatter plot will be saved to:\n"+ \
                        os.path.join(self.output_path, "./som_projected.png"))

    def plot_clusters(self, coor, clusters,
                             color_val=None,
                             project=False, jitter=False, 
                             show=False, print_out=True,
                             **kwargs):

        """Project points onto the trained 2D map and plot the result.

        Args:
            coor (array): An array containing datapoints to be mapped or
                pre-mapped if project False.
            clusters (list): Cluster assignment list.
            color_val (array): The feature value to use as color map, if None
                the map will be plotted as white.
            project (bool): if True, project the points in coor onto the map.
            jitter (bool): if True, add jitter to points coordinates to help
                with overlapping points.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot:
                - figsize (tuple(int, int)): the figure size,
                - title (str): figure title,
                - cbar_label (str): colorbar label,
                - labelsize (int): font size of label, 
                    the title will be 15% larger,
                    ticks will be 15% smaller.

        """
        
        bmu_coor = self.project_onto_map(coor) if project else coor

        if jitter:
            bmu_coor += np.random.uniform(low=-.15, high=.15, size=(bmu_coor.shape[0],2))

        _, _ = scatter_on_map([bmu_coor[clusters==clu] for clu in set(clusters)], 
                       [[node.pos[0],node.pos[1]] for node in self.node_list],
                       self.polygons,
                       color_val=color_val, 
                       show=show, print_out=print_out,
                       file_name=os.path.join(self.output_path, "./som_clusters.png"),
                       **kwargs)

        if print_out:
            logger.info("Clustering plot will be saved to:\n"+ \
                        os.path.join(self.output_path, "./som_clustering.png"))

        
class SOMNode:
    """ Single Kohonen SOM node class. """
    
    def __init__(self, x, y, num_weights, net_height, net_width, PBC, polygons,
                weight_bounds=None, init_vec=None, weights_array=None):
    
        """Initialize the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            num_weights (int): Length of the weights vector.
            net_height (int): Network height, needed for periodic boundary conditions (PBC)
            net_width (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            polygons (Polygon obj): a polygon object with information on the map topology.
            weight_bounds(array): boundary values for the random initialization
                of the weights. Must be in the format [min_val, max_val]. 
                They are overwritten by "init_vec".
            init_vec (array): Array containing the two custom vectors (e.g. PCA)
                for the weights initalization.
            weights_array (array): Array containing the weights to give
                to the node if loaded from a file.
        """
        
        self.polygons  = polygons
        self.PBC       = PBC
        
        self.pos = polygons.to_tiles((x,y))

        self.weights    = []
        self.difference = None 

        self.net_height = net_height
        self.net_width  = net_width

        if weights_array is not None:            
            self.weights = weights_array

        elif init_vec is not None:
            # Sample uniformly in the space spanned by the custom vectors.
            self.weights = ((x-self.net_width/2)*2.0/self.net_width*init_vec[0] + 
                            (y-self.net_height/2)*2.0/self.net_height*init_vec[1])

        elif weight_bounds is not None:
            #Sample Select randomly in the space spanned by the data. 
            for i in range(num_weights):
                self.weights.append(random.random()*(weight_bounds[1][i]-weight_bounds[0][i])+weight_bounds[0][i])
       
        else: 
            logger.error("Error in the network weights initialization, make sure to provide random initalization boundaries,\n"+ \
                         "custom vectors, or load the weights from file.")
            sys.exit(1)
   
        self.weights = np.array(self.weights)

    def get_node_distance(self, node):
        """ Calculate the distance within the network between the current node and second node.

        Args:
            node (SOMNode): The node from which the distance is calculated.
            
        Returns:
            (float): The distance between the two nodes.
        """

        if self.PBC:
            return self.polygons.distance_pbc((self.pos, node.pos), 
                                        (self.net_width, self.net_height),
                                         lambda x, y: np.sqrt(np.sum(np.square(x-y))))
        else:
            return np.sum(np.square(self.pos - node.pos))

    def _update_weights(self, input_vec, sigma, learning_rate, bmu):
        """ Update the node weights.

        Args:
            input_vec (array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            learning_rate (float): The updated learning rate.
            bmu (SOMNode): The best matching unit.
        """
    
        dist  = self.get_node_distance(bmu)
        gauss = np.exp(-dist**2/(2*sigma**2))
        
        self.weights -= gauss*learning_rate*(self.weights-input_vec)
        
        
    def _set_difference(self, diff_value):
        """ Set the neighbouring nodes weights difference."""

        self.difference = np.float(diff_value)


if __name__ == "__main__":

    pass