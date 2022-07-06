"""
Pytests.

F Comitani, SG Riva, A Tangherloni 
"""

import os
import shutil

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import simpsom as sps

from sklearn.datasets import load_digits

class Parameters:

    output_path = os.path.join("newtork_test")
    truth_path = os.path.join("ground_truth")

    GPU=[False]
    try:
        import cupy
        GPU += [True]
    except ImportError:
        pass

class TestNetwork:

    @classmethod
    def setup_method(cls, capsys):
        
        if os.path.exists(Parameters.output_path):
            shutil.rmtree(Parameters.output_path)
        os.mkdir(Parameters.output_path)

        if True not in Parameters.GPU:
            print('No CuPy found, GPU tests will be skipped.')

    @classmethod
    def teardown_method(cls):

       shutil.rmtree(Parameters.output_path)

    @staticmethod
    @pytest.fixture()
    def load_dataset():

        digits = load_digits()
        return digits.data

    @pytest.mark.parametrize("PBC,size,init,\
                              metric,topology,neighborhood_fun,\
                              train_algo,epochs,early_stop,\
                              plotall,load", [
    (False, 10, 'random', 'euclidean', 'hexagonal', 'gaussian', 'online', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian', 'online', 10, None, False, True),
    (False, 10, 'PCA', 'cosine', 'hexagonal', 'gaussian', 'online', 10, None, False, False),
    (False, 10, 'PCA', 'manhattan', 'hexagonal', 'gaussian', 'online', 10, None, False, False),
    (True, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian', 'online', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian', 'batch', 10, None, False, False),
    (False, 10, 'PCA', 'cosine', 'hexagonal', 'gaussian', 'batch', 10, None, False, False),
    (False, 10, 'PCA', 'manhattan', 'hexagonal', 'gaussian', 'batch', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'mexican_hat', 'batch', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'bubble', 'batch', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian', 'batch', 100, 'mapdiff', False, False),
    (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian', 'batch', 100, 'bmudiff', False, False),
    (False, 10, 'PCA', 'euclidean', 'square', 'gaussian', 'online', 10, None, False, False),
    (False, 10, 'PCA', 'euclidean', 'square', 'gaussian', 'batch', 10, 'mapdiff', True, False)
    ])
    @pytest.mark.parametrize("GPU", Parameters.GPU)
    def test_som(self, load_dataset, PBC, size, init,
                 metric, topology, neighborhood_fun,
                 train_algo, epochs, early_stop,
                 plotall, load, GPU):

        data = load_dataset
        hashed_name = int.from_bytes((str(PBC)+str(init)+
               str(metric)+str(topology)+str(neighborhood_fun)+\
               str(train_algo)+str(epochs)+str(early_stop)).encode(), 'little')
        
        net = sps.SOMNet(size, size, data, 
                         topology=topology, PBC=PBC, 
                         init=init, metric=metric, 
                         random_seed=32, GPU=GPU, debug=False, 
                         output_path=Parameters.output_path)
        net.train(train_algo=train_algo, start_learning_rate=0.01, epochs=epochs, 
                  batch_size=-1, early_stop=early_stop)

        net.save_map("trained_som_{:d}.npy".format(hashed_name))
        assert(os.path.isfile(os.path.join(Parameters.output_path, "trained_som_{:d}.npy".format(hashed_name))))
        shutil.copyfile(os.path.join(Parameters.output_path, "trained_som_{:d}.npy".format(hashed_name)), 
                        os.path.join(Parameters.truth_path, "trained_som_{:d}.npy".format(hashed_name)))
        assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "trained_som_{:d}.npy".format(hashed_name)), allow_pickle=True),
                           np.load(os.path.join(Parameters.truth_path, "trained_som_{:d}.npy".format(hashed_name)), allow_pickle=True), decimal=4)

        if load:
            net_l = sps.SOMNet(size, size, data, 
                         load_file=os.path.join(Parameters.output_path, "trained_som_{:d}.npy".format(hashed_name)),
                         topology=topology, PBC=PBC, 
                         init=init, metric=metric, 
                         random_seed=32, GPU=GPU, debug=False, 
                         output_path=Parameters.output_path)

            weights_array = [[float(net_l.net_height)]*net_l.nodes_list[0].weights.shape[0], 
                             [float(net_l.net_width)]*net_l.nodes_list[0].weights.shape[0],
                             [float(net_l.PBC)]*net_l.nodes_list[0].weights.shape[0]] + \
                            [net_l._get(node.weights) for node in net_l.nodes_list]
            assert_array_equal(np.array(weights_array),
                np.load(os.path.join(Parameters.output_path, "trained_som_{:d}.npy".format(hashed_name)), allow_pickle=True))
            
        if plotall:

            net.plot_map_by_feature(feature=1, show=False, print_out=True)
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_feature_1.png")))

            net.plot_map_by_difference(show=False, print_out=True, returns=False)
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_difference.png")))

            net.plot_projected_points(net.project_onto_map(data), 
                                        project=False, jitter=True, 
                                        show=False, print_out=True)
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_projected.png")))

            net.plot_projected_points(data, color_val=[n.difference for n in net.nodes_list],
                                        project=True, jitter=True, 
                                        show=False, print_out=True, 
                                        file_name=os.path.join(Parameters.output_path, "som_projected_2.png"))
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_projected_2.png")))

            labs, points = net.cluster(data, algorithm='AgglomerativeClustering', file_name="som_clusters_AC.npy")
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_projected_AgglomerativeClustering.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_projected_AgglomerativeClustering.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_projected_AgglomerativeClustering.npy"), allow_pickle=True), decimal=4)

            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_AC.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_clusters_AC.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_clusters_AC.npy"), allow_pickle=True), decimal=4)

            net.plot_clusters(data, labs, project=True, show=False, print_out=True,
                              file_name=os.path.join(Parameters.output_path, "som_clusters_AC.png"))
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_AC.png")))

            labs, points = net.cluster(data, algorithm='DBSCAN', file_name="som_clusters_DB.npy")
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_projected_DBSCAN.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_projected_DBSCAN.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_projected_DBSCAN.npy"), allow_pickle=True), decimal=4)

            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_DB.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_clusters_DB.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_clusters_DB.npy"), allow_pickle=True), decimal=4)

            net.plot_clusters(data, labs, project=True, show=False, print_out=True,
                              file_name=os.path.join(Parameters.output_path, "som_clusters_DB.png"))
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_DB.png")))

            labs, points = net.cluster(data, algorithm='KMeans', file_name="som_clusters_KM.npy", random_state=32)
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_projected_KMeans.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_projected_KMeans.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_projected_KMeans.npy"), allow_pickle=True), decimal=4)

            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_KM.npy")))
            assert_array_almost_equal(np.load(os.path.join(Parameters.output_path, "som_clusters_KM.npy"), allow_pickle=True),
                               np.load(os.path.join(Parameters.truth_path, "som_clusters_KM.npy"), allow_pickle=True), decimal=4)

            net.plot_clusters(data, labs, project=True, show=False, print_out=True,
                              file_name=os.path.join(Parameters.output_path, "som_clusters_KM.png"))
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_clusters_KM.png")))


            net.plot_convergence(fsize=(5, 5), logax=False, show=False, print_out=True)
            assert(os.path.isfile(os.path.join(Parameters.output_path, "som_convergence.png")))



if __name__ == '__main__':
    pass