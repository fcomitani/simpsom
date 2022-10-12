from typing import Union, List, Tuple
from types import ModuleType

import sys
import os
import subprocess
import multiprocessing

from functools import partial

import numpy as np

from loguru import logger

from simpsom.distances import Distance
from simpsom.neighborhoods import Neighborhoods
from simpsom.polygons import Squares, Hexagons, Polygon
from simpsom.plots import plot_map, line_plot, scatter_on_map
from simpsom.early_stop import EarlyStop


class SOMNet:
    """ Kohonen SOM Network class. """

    def __init__(self, net_height: int, net_width: int, data: np.ndarray,
                 load_file: str = None, metric: str = "euclidean",
                 topology: str = "hexagonal", neighborhood_fun: str = "gaussian",
                 init: str = "random", PBC: bool = False,
                 GPU: bool = False, CUML: bool = False,
                 random_seed: int = None, debug: bool = False, output_path: str = "./") -> None:
        """ Initialize the SOM network.

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
                or PCA (default "random").
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

        self.GPU = bool(GPU)
        self.CUML = bool(CUML)

        if self.GPU:
            try:
                import cupy
                self.xp = cupy

                if self.CUML:
                    try:
                        from cuml import cluster
                    except:
                        logger.warning(
                            "CUML libraries not found. Scikit-learn will be used instead.")

            except:
                logger.warning(
                    "CuPy libraries not found. Falling back to CPU.")
                self.GPU = False

        try:
            self.xp
        except:
            self.xp = np

        try:
            cluster
        except:
            from sklearn import cluster
        self.cluster_algo = cluster

        if random_seed is not None:
            os.environ["PYTHONHASHSEED"] = str(random_seed)
            np.random.seed(random_seed)
            self.xp.random.seed(random_seed)

        self.PBC = bool(PBC)
        if self.PBC:
            logger.info("Periodic Boundary Conditions active.")

        self.nodes_list = []
        self.data = self.xp.array(data, dtype=np.float32)

        self.metric = metric

        if topology.lower() == "hexagonal":
            self.polygons = Hexagons
            logger.info("Hexagonal topology.")
        else:
            self.polygons = Squares
            logger.info("Square topology.")

        self.distance = Distance(self.xp)

        self.neighborhood_fun = neighborhood_fun.lower()
        if self.neighborhood_fun not in ['gaussian', 'mexican_hat', 'bubble']:
            logger.error("{} neighborhood function not recognized.".format(self.neighborhood_fun) +
                         "Choose among 'gaussian', 'mexican_hat' or 'bubble'.")
            raise ValueError
        self.neighborhoods = Neighborhoods(self.xp)

        self.convergence = []

        self.net_height = net_height
        self.net_width = net_width

        self.init = init
        if isinstance(self.init, str):
            self.init = self.init.lower()
        else:
            self.init = self.xp.array(self.init)
        self._set_weights(load_file)

    def _get(self, data: np.ndarray) -> np.ndarray:
        """ Moves data from GPU to CPU.
        If already on CPU, it will be left as it is.

        Args:
            data (array): data to move from GPU to CPU.

        Returns:
            (array): the same data on CPU.
        """

        if self.xp.__name__ == "cupy":
            if isinstance(data, list):
                return [d.get() for d in data]
            elif isinstance(data, np.ndarray):
                return data
            return data.get()

        return data

    def _set_weights(self, load_file: str = None) -> None:
        """ Set initial map weights values, either by loading them from file or with random/PCA.

        Args:
            load_file (str): Name of file to load containing information
                to initialize the network weights.
        """

        init_vec = None
        weights_array = None
        this_weight = None

        # When loaded from file, element 0 contains information on the network shape
        count_weight = 3

        if load_file is None:

            if isinstance(self.init, str) and self.init == "pca":
                logger.warning(
                    "Please be sure that the data have been standardized before using PCA.")
                logger.info("The weights will be initialized with PCA.")

                if self.GPU:
                    matrix = self.data.get()
                    init_vec = self.pca(matrix, n_pca=2)
                    init_vec = self.xp.array(init_vec)
                else:
                    matrix = self.data
                    init_vec = self.pca(matrix, n_pca=2)

            else:
                logger.info("The weights will be initialized randomly.")
                init_vec = [self.xp.min(self.data, axis=0),
                            self.xp.max(self.data, axis=0)]

        else:
            logger.info("The weights will be loaded from file.\n" +
                        "The map shape will be overwritten and no weights" +
                        "initialization will be applied.")
            if not load_file.endswith(".npy"):
                load_file += ".npy"
            weights_array = np.load(load_file, allow_pickle=True)
            self.net_height = int(weights_array[0][0])
            self.net_width = int(weights_array[1][0])
            self.PBC = bool(weights_array[2][0])

        for x in range(self.net_width):
            for y in range(self.net_height):

                if weights_array is not None:
                    this_weight = weights_array[count_weight]
                    count_weight += 1

                self.nodes_list.append(SOMNode(x, y, self.data.shape[1],
                                               self.net_height, self.net_width,
                                               self.PBC, self.polygons,
                                               self.xp,
                                               init_vec=init_vec,
                                               weights_array=this_weight))

    def pca(self, matrix: np.ndarray, n_pca: int) -> np.ndarray:
        """Get principal components to initialize network weights.

        Args:
            matrix (array): N-dimensional dataset.
            n_pca (int): number of components to keep.

        Returns:
            (array): Principal axes in feature space,
                representing the directions of maximum variance in the data.
        """

        mean_vector = np.mean(matrix.T, axis=1)
        center_mat = matrix - mean_vector
        cov_mat = np.cov(center_mat.T).astype(self.xp.float32)

        return np.linalg.eig(cov_mat)[-1].T[:n_pca]

    def _get_n_process(self) -> int:
        """ Count number of GPU or CPU processors.

        Returns:
            (int): the number of processors.
        """

        if self.xp.__name__ == "cupy":
            try:
                dev = self.xp.cuda.Device()
                n_smp = dev.attributes["MultiProcessorCount"]
                max_thread_per_smp = dev.attributes["MaxThreadsPerMultiProcessor"]
                return n_smp * max_thread_per_smp
            except:
                logger.error("Something went wrong when trying to count the number\n" +
                             "of GPU processors from CuPy.")
                return -1

        else:
            try:
                return multiprocessing.cpu_count()
            except:
                logger.error("Something went wrong when trying to count the number\n" +
                             "of CPU processors.")
                return -1

    def _randomize_dataset(self, data: np.ndarray, epochs: int) -> np.ndarray:
        """Generates a random list of datapoints indices for online training.

        Args:
            data (array or list): N-dimensional dataset.
            epochs (int): Number of training iterations.

        Returns:
            entries (array): array with randomized indices
        """

        if epochs < data.shape[0]:
            logger.warning(
                "Epochs for online training are less than the input datapoints.")
            epochs = data.shape[0]

        iterations = int(np.ceil(epochs/data.shape[0]))

        return [ix
                for shuffled in [np.random.permutation(data.shape[0])
                                 for it in np.arange(iterations)]
                for ix in shuffled]

    def save_map(self, file_name: str = "trained_som.npy") -> None:
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            file_name (str): Name of the file where the data will be saved.
        """

        weights_array = [[float(self.net_height)]*self.nodes_list[0].weights.shape[0],
                         [float(self.net_width)] *
                         self.nodes_list[0].weights.shape[0],
                         [float(self.PBC)]*self.nodes_list[0].weights.shape[0]] + \
            [self._get(node.weights) for node in self.nodes_list]

        if not file_name.endswith((".npy")):
            file_name += ".npy"
        logger.info("Map shape and weights will be saved to:\n" +
                    os.path.join(self.output_path, file_name))
        np.save(os.path.join(self.output_path, file_name),
                np.array(weights_array))

    def _update_sigma(self, n_iter: int) -> None:
        """Update the gaussian sigma.

        Args:
            n_iter (int): Iteration number.
        """

        self.sigma = self.start_sigma * self.xp.exp(-n_iter/self.tau)

    def _update_learning_rate(self, n_iter: int) -> None:
        """Update the learning rate.

        Args:
            n_iter (int): Iteration number.
        """

        self.learning_rate = self.start_learning_rate * \
            self.xp.exp(n_iter/self.epochs)

    def find_bmu_ix(self, vecs: np.array) -> 'SOMNode':
        """Find the index of the best matching unit (BMU) for a given list of vectors.

        Args:
            vec (array or list[lists, ..]): vectors whose distance from the network
                nodes will be calculated.

        Returns:
            bmu (SOMNode): The best matching unit node index.
        """

        dists = self.distance.pairdist(vecs,
                                       self.xp.array(
                                           [n.weights for n in self.nodes_list]),
                                       metric=self.metric)

        return self.xp.argmin(dists, axis=1)

    # TODO: Consider changing epoch in online training to be equivalent to 1 full run
    # of the dataset rather than a single data point
    def train(self, train_algo: str = "batch", epochs: int = -1,
              start_learning_rate: float = 0.01, early_stop: str = None,
              early_stop_patience: int = 3, early_stop_tolerance: float = 1e-4, batch_size: int = -1) -> None:
        """Train the SOM.

        Args:
            train_algo (str): training algorithm, choose between "online" or "batch"
                (default "online"). Beware that the online algorithm will run one datapoint
                per epoch, while the batch algorithm runs all points at one for each epoch.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints.
                Warning: for online training each epoch corresponds to 1 sample in the
                input dataset, for batch training it corresponds to one full dataset
                training.
            start_learning_rate (float): Initial learning rate, used only in online
                learning.
            early_stop (str): Early stopping method, for now only "mapdiff" (checks if the
                weights of nodes don"t change) is available. If None, don"t use early stopping (default None).
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

        logger.info("The map will be trained with the " +
                    train_algo+" algorithm.")
        self.start_sigma = max(self.net_height, self.net_width)/2
        self.start_learning_rate = start_learning_rate

        self.data = self.xp.array(self.data)

        if epochs == -1:
            if train_algo == 'online':
                epochs = self.data.shape[0]*10
            else:
                epochs = 10

        self.epochs = epochs
        self.tau = self.epochs/self.xp.log(self.start_sigma)

        if early_stop not in ["mapdiff", None]:
            logger.warning("Convergence method not recognized, early stopping will be deactivated. " +
                           "Currently only \"mapdiff\" is available.")
            early_stop = None

        if early_stop is not None:
            logger.info("Early stop active.")
            logger.warning("Early stop is an experimental feature, " +
                           "make sure to know what you are doing!")

        early_stopper = EarlyStop(tolerance=early_stop_tolerance,
                                  patience=early_stop_patience)

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

                if early_stopper.stop_training:
                    logger.info(
                        "\rEarly stop tolerance reached at epoch {:d}, training will be stopped.".format(n_iter-1))
                    self.convergence = early_stopper.convergence
                    break

                if n_iter % 10 == 0:
                    logger.debug("\rTraining SOM... {:d}%".format(
                        int(n_iter*100.0/self.epochs)))

                self._update_sigma(n_iter)
                self._update_learning_rate(n_iter)

                datapoint_ix = datapoints_ix.pop()
                input_vec = self.data[datapoint_ix, :].reshape(
                    1, self.data.shape[1])

                bmu = self.nodes_list[int(self.find_bmu_ix(input_vec)[0])]

                for node in self.nodes_list:
                    node._update_weights(
                        input_vec[0], self.sigma, self.learning_rate, bmu)

                if n_iter % self.data.shape[0] == 0 and early_stop is not None:
                    early_stopper.check_convergence(
                        early_stopper.calc_loss(self))

        elif train_algo == "batch":
            """ Batch training.
            All datapoints are used at once for each epoch,
            the weights are updated with the sum of contributions from all these points.
            No learning rate needed.

            Kinouchi, M. et al. "Quick Learning for Batch-Learning Self-Organizing Map" (2002).
            """

            # Storing the distances and weight matrices defeats the purpose of having
            # nodes as instances of a class, but it helps with the optimization
            # and parallelization at the cost of memory.
            # The object-oriented structure is kept to simplify code reading.

            all_weights = self.xp.array([n.weights for n in self.nodes_list])
            all_weights = all_weights.reshape(
                self.net_width, self.net_height, self.data.shape[1])

            numerator = self.xp.zeros(all_weights.shape, dtype=self.xp.float32)
            denominator = self.xp.zeros(
                (all_weights.shape[0], all_weights.shape[1], 1), dtype=self.xp.float32)

            unravel_precomputed = self.xp.unravel_index(self.xp.arange(self.net_width*self.net_height),
                                                        (self.net_width, self.net_height))

            _xx, _yy = self.xp.meshgrid(self.xp.arange(
                self.net_width), self.xp.arange(self.net_height))

            neighborhood_caller = partial(
                self.neighborhoods.neighborhood_caller, xx=_xx, yy=_yy,
                neigh_func=self.neighborhood_fun,
                pbc_func=self.polygons.neighborhood_pbc if self.PBC
                else None)

            for n_iter in range(self.epochs):

                if early_stopper.stop_training:
                    logger.info(
                        "\rEarly stop tolerance reached at epoch {:d}, training will be stopped.".format(n_iter-1))
                    self.convergence = early_stopper.convergence
                    break

                self._update_sigma(n_iter)
                self._update_learning_rate(n_iter)

                if n_iter % 10 == 0:
                    logger.debug("Training SOM... {:.2f}%".format(
                        n_iter*100.0/self.epochs))

                try:
                    numerator.fill(0)
                    denominator.fill(0)
                except AttributeError:
                    numerator = self.xp.zeros(all_weights.shape)
                    denominator = self.xp.zeros(
                        (all_weights.shape[0], all_weights.shape[1], 1))

                for i in range(0, len(self.data), _n_parallel):
                    start = i
                    end = start + _n_parallel
                    if end > len(self.data):
                        end = len(self.data)

                    batchdata = self.data[start:end]

                    # Find BMUs for all points and subselect gaussian matrix.
                    dists = self.distance.batchpairdist(
                        batchdata, all_weights, self.metric)

                    raveled_idxs = dists.argmin(axis=1)
                    wins = (
                        unravel_precomputed[0][raveled_idxs], unravel_precomputed[1][raveled_idxs])

                    g_gpu = neighborhood_caller(
                        wins, sigma=self.sigma)*self.learning_rate

                    sum_g_gpu = self.xp.sum(g_gpu, axis=0)
                    g_flat_gpu = g_gpu.reshape(g_gpu.shape[0], -1)
                    gT_dot_x_flat_gpu = self.xp.dot(g_flat_gpu.T, batchdata)

                    numerator += gT_dot_x_flat_gpu.reshape(numerator.shape)
                    denominator += sum_g_gpu[:, :, self.xp.newaxis]

                new_weights = self.xp.where(
                    denominator != 0, numerator / denominator, all_weights)

                if early_stop is not None:
                    loss = self.xp.abs(self.xp.subtract(
                        new_weights, all_weights)).mean()
                    print(loss)
                    early_stopper.check_convergence(loss)

                all_weights = new_weights

            # Revert to object oriented
            all_weights = all_weights.reshape(
                self.net_width*self.net_height, self.data.shape[1])
            for n_node, node in enumerate(self.nodes_list):
                node.weights = all_weights[n_node]  # * self.learning_rate

        else:
            logger.error(
                "Training algorithm not recognized. Choose between \"online\" and \"batch\".")
            sys.exit(1)

        if self.GPU:
            for node in self.nodes_list:
                node.weights = node.weights.get()
        if early_stop is not None:
            self.convergence = [arr.get(
            ) for arr in early_stopper.convergence] if self.GPU else early_stopper.convergence

    def get_nodes_difference(self) -> None:
        """ Extracts the neighbouring nodes difference in weights and assigns it
        to each node object.
        """

        weights = self.xp.array([node.weights for node in self.nodes_list])
        pos = self.xp.array([node.pos for node in self.nodes_list])
        weights_dist = self.distance.pairdist(
            weights, weights, metric=self.metric)

        # if self.PBC:
        # TODO: a precision issue with the PBC nodes position creates an ugly line
        # at the top and bottom of the map.
        #    pos_dist = self.polygons.distance_pbc(pos, pos,
        #        net_shape=(self.net_width,self.net_height),
        #        distance_func=lambda x, y: self.distance.pairdist(x, y, metric='euclidean'),
        #        xp=self.xp,
        #        axis=0)
        # else:
        pos_dist = self.distance.pairdist(pos, pos, metric='euclidean')

        weights_dist[(pos_dist > 1.01) | (pos_dist == 0.)] = np.nan
        [node._set_difference(value)
            for node, value in zip(self.nodes_list, self.xp.nanmean(weights_dist, axis=0))]

        logger.info('Weights difference among neighboring nodes calculated.')

    def project_onto_map(self, array: np.ndarray,
                         file_name: str = "./som_projected.npy") -> list:
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
            array = self.xp.array(array)

        bmu_list = [
            self.nodes_list[int(mu)].pos for mu in self.find_bmu_ix(array)]

        if file_name is not None:
            if not file_name.endswith((".npy")):
                file_name += ".npy"
            logger.info("Projected coordinates will be saved to:\n" +
                        os.path.join(self.output_path, file_name))
            np.save(os.path.join(self.output_path,
                    file_name), self._get(bmu_list))

        return self.xp.array(bmu_list)

    def cluster(self, coor: np.ndarray, project: bool = True, algorithm: str = "DBSCAN",
                file_name: str = "./som_clusters.npy", **kwargs: str) -> List[int]:
        """Project data onto the map and find clusters with scikit-learn clustering algorithms.

        Args:
            coor (array): An array containing datapoints to be mapped or
                pre-mapped if project False.
            project (bool): if True, project the points in coor onto the map.
            algorithm (clustering obj or str): The clusters identification algorithm. A scikit-like
                class can be provided (must have a fit method), or a string indicating one of the algorithms
                provided by the scikit library
            file_name (str): Name of the file to which the data will be saved
                if not None.
            kwargs (dict): Keyword arguments to the clustering algorithm:

            Returns:
            (list of int): A list containing the clusters of the input array datapoints.
        """

        bmu_coor = self.project_onto_map(coor, file_name="som_projected_"+algorithm+".npy") \
            if project else coor
        if self.xp.__name__ == "cupy" and self.cluster_algo.__name__.startswith('sklearn'):
            bmu_coor = self._get(bmu_coor)

        if self.PBC:
            # Implementing the distance_pbc as a wrapper automatically applied to the provided metric
            # is not possible as many sklearn clustering functions don't allow for custom metric.
            logger.warning("PBC are active. Make sure to provide a PBC-compatible custom metric if possible, " +
                           "or use `polygons.distance_pbc`. See the documentation for more detail.")

        if type(algorithm) is str:

            import inspect
            modules = [module[0] for module in inspect.getmembers(
                self.cluster_algo, inspect.isclass)]

            if algorithm not in modules:
                logger.error("The desired algorithm is not among the algorithms provided by the scikit library,\n" +
                             "please provide one of the algorithms provided by the scikit library:\n" +
                             "|".join(modules))
                return None, None

            clu_algo = eval("self.cluster_algo."+algorithm)

        else:
            clu_algo = algorithm

            if not callable(getattr(clu_algo, "fit", None)):
                logger.error("There was a problem with the clustering, make sure to provide a scikit-like clustering\n" +
                             "class or use one of the algorithms provided by the scikit library,\n" +
                             "Custom classes must have a 'fit' method.")
                return None, None

        clu_algo = clu_algo(**kwargs)

        try:
            clu_labs = clu_algo.fit(bmu_coor).labels_
        except:
            logger.error("There was a problem with the clustering, make sure to provide a scikit-like clustering\n" +
                         "class or use one of the algorithms provided by the scikit library,\n" +
                         "Custom classes must have a 'fit' method.")
            return None

        if file_name is not None:
            if not file_name.endswith((".npy")):
                file_name += ".npy"
            logger.info("Clustering results will be saved to:\n" +
                        os.path.join(self.output_path, file_name))
            np.save(os.path.join(self.output_path,
                    file_name), self._get(clu_labs))

        return clu_labs, bmu_coor

    def plot_map_by_feature(self, feature: int, show: bool = False, print_out: bool = True,
                            **kwargs: Tuple[int]) -> None:
        """ Wrapper function to plot a trained 2D SOM map
        color-coded according to a given feature.

        Args:
            feature (int): The feature number to use as color map.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot, such as 
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        if "file_name" not in kwargs.keys():
            kwargs["file_name"] = os.path.join(self.output_path,
                                               "./som_feature_{}.png".format(str(feature)))

        _, _ = plot_map([[node.pos[0], node.pos[1]] for node in self.nodes_list],
                        [node.weights[feature] for node in self.nodes_list],
                        self.polygons,
                        show=show, print_out=print_out,
                        **kwargs)

        if print_out:
            logger.info("Feature map will be saved to:\n" +
                        kwargs["file_name"])

    def plot_map_by_difference(self, show: bool = False, print_out: bool = True,
                               **kwargs: Tuple[int]) -> None:
        """ Wrapper function to plot a trained 2D SOM map
        color-coded according neighbours weights difference.
        It will automatically calculate the difference values
        if not already computed.

        Args:
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        if "file_name" not in kwargs.keys():
            kwargs["file_name"] = os.path.join(
                self.output_path, "./som_difference.png")

        if self.nodes_list[0].difference is None:
            self.get_nodes_difference()

        if "cbar_label" not in kwargs.keys():
            kwargs["cbar_label"] = "Nodes difference value"

        _, _ = plot_map([[node.pos[0], node.pos[1]] for node in self.nodes_list],
                        [node.difference for node in self.nodes_list],
                        self.polygons,
                        show=show, print_out=print_out,
                        **kwargs)

        if print_out:
            logger.info("Node difference map will be saved to:\n" +
                        kwargs["file_name"])

    def plot_convergence(self, show: bool = False, print_out: bool = True,
                         **kwargs: Tuple[int]) -> None:
        """ Plot the the map training progress according to the
        chosen convergence criterion, when train_algo is batch.

        Args:
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - xlabel (str), x-axis label;
                - ylabel (str), y-axis label;
                - logx (bool), if True set x-axis to logarithmic scale;
                - logy (bool), if True set y-axis to logarithmic scale;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
        """

        if len(self.convergence) == 0:
            logger.warning(
                "The current parameters yelded no convergence. The plot will not be produced.")

        else:

            if "file_name" not in kwargs.keys():
                kwargs["file_name"] = os.path.join(
                    self.output_path, "./som_convergence.png")

            conv_values = np.nan_to_num(self.convergence)

            if "title" not in kwargs.keys():
                kwargs["title"] = "Convergence"
            if "xlabel" not in kwargs.keys():
                kwargs["xlabel"] = "Iteration"
            if "ylabel" not in kwargs.keys():
                kwargs["ylabel"] = "Score"

            _, _ = line_plot(conv_values,
                             show=show, print_out=print_out,
                             **kwargs)

            if print_out:
                logger.info("Convergence results will be saved to:\n" +
                            kwargs["file_name"])

    def plot_projected_points(self, coor: np.ndarray, color_val: Union[np.ndarray, None] = None,
                              project: bool = True, jitter: bool = True,
                              show: bool = False, print_out: bool = True,
                              **kwargs: Tuple[int]) -> None:
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
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        if "file_name" not in kwargs.keys():
            kwargs["file_name"] = os.path.join(
                self.output_path, "./som_projected.png")

        bmu_coor = self.project_onto_map(coor) if project else coor
        bmu_coor = self._get(bmu_coor)

        if jitter:
            bmu_coor = np.array(bmu_coor).astype(float)
            bmu_coor += np.random.uniform(low=-.15,
                                          high=.15, size=(bmu_coor.shape[0], 2))

        _, _ = scatter_on_map([bmu_coor],
                              [[node.pos[0], node.pos[1]]
                              for node in self.nodes_list],
                              self.polygons,
                              color_val=color_val,
                              show=show, print_out=print_out,
                              **kwargs)

        if print_out:
            logger.info("Projected data scatter plot will be saved to:\n" +
                        kwargs["file_name"])

    def plot_clusters(self, coor: np.ndarray, clusters: list,
                      color_val: np.ndarray = None,
                      project: bool = False, jitter: bool = False,
                      show: bool = False, print_out: bool = True,
                      **kwargs: Tuple[int]) -> None:
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
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        if "file_name" not in kwargs.keys():
            kwargs["file_name"] = os.path.join(
                self.output_path, "./som_clusters.png")

        bmu_coor = self.project_onto_map(coor) if project else coor
        bmu_coor = self._get(bmu_coor)

        if jitter:
            bmu_coor += np.random.uniform(low=-.15,
                                          high=.15, size=(bmu_coor.shape[0], 2))

        _, _ = scatter_on_map([bmu_coor[clusters == clu] for clu in set(clusters)],
                              [[node.pos[0], node.pos[1]]
                              for node in self.nodes_list],
                              self.polygons,
                              color_val=color_val,
                              show=show, print_out=print_out,
                              **kwargs)

        if print_out:
            logger.info("Clustering plot will be saved to:\n" +
                        kwargs["file_name"])


class SOMNode:
    """ Single Kohonen SOM node class. """

    def __init__(self, x: int, y: int, num_weights: int, net_height: int, net_width: int,
                 PBC: bool, polygons: Polygon, xp: ModuleType = np,
                 init_vec: Union[np.ndarray, None] = None,
                 weights_array: Union[np.ndarray, None] = None) -> None:
        """Initialize the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            num_weights (int): Length of the weights vector.
            net_height (int): Network height, needed for periodic boundary conditions (PBC)
            net_width (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            polygons (Polygon obj): a polygon object with information on the map topology.
            xp (numpy or cupy): the numeric library to be used.
            weight_bounds(array): boundary values for the random initialization
                of the weights. Must be in the format [min_val, max_val].
                They are overwritten by "init_vec".
            init_vec (array): Array containing the two custom vectors (e.g. PCA)
                for the weights initalization.
            weights_array (array): Array containing the weights to give
                to the node if loaded from a file.
        """

        self.xp = xp
        self.polygons = polygons
        self.PBC = PBC

        self.pos = self.xp.array(polygons.to_tiles((x, y)))

        self.weights = []
        self.difference = None

        self.net_height = net_height
        self.net_width = net_width

        if weights_array is not None:
            self.weights = weights_array

        elif init_vec is not None:
            # Sample uniformly in the space spanned by the custom vectors.
            self.weights = (init_vec[1] - init_vec[0])*self.xp.array(
                np.random.rand(len(init_vec[0])).astype(np.float32)) + init_vec[0]

        else:
            logger.error("Error in the network weights initialization, make sure to provide random initalization boundaries,\n" +
                         "custom vectors, or load the weights from file.")
            sys.exit(1)

        self.weights = self.xp.array(self.weights)

    def get_node_distance(self, node: 'SOMNode') -> float:
        """ Calculate the distance within the network between the current node and second node.

        Args:
            node (SOMNode): The node from which the distance is calculated.

        Returns:
            (float): The distance between the two nodes.
        """

        if self.PBC:
            return self.polygons.distance_pbc(self.pos, node.pos,
                                              (self.net_width, self.net_height),
                                              lambda x, y: self.xp.sqrt(
                                                  self.xp.sum(self.xp.square(x-y))),
                                              xp=self.xp)
        else:
            return self.xp.sqrt(self.xp.sum(self.xp.square(self.pos - node.pos)))

    def _update_weights(self, input_vec: np.ndarray, sigma: float, learning_rate: float, bmu: 'SOMNode') -> None:
        """ Update the node weights.

        Args:
            input_vec (array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            learning_rate (float): The updated learning rate.
            bmu (SOMNode): The best matching unit.
        """

        dist = self.get_node_distance(bmu)
        gauss = self.xp.exp(-dist**2/(2*sigma**2))

        self.weights -= gauss*learning_rate*(self.weights-input_vec)

    def _set_difference(self, diff_value: Union[float, int]) -> None:
        """ Set the neighbouring nodes weights difference.

        Args:
            diff_value (float or int), the difference value to set.
        """

        self.difference = self.xp.float(diff_value)


if __name__ == "__main__":

    pass
