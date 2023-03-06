import os
import platform
import shutil
import hashlib
import base64

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.datasets import load_digits

import simpsom as sps


class Parameters:
    output_path = os.path.join("tests/newtork_test")
    truth_path = os.path.join("tests/ground_truth")

    GPU = [False]
    try:
        import cupy
        GPU += [True]
    except ImportError:
        pass


class TestNetwork:
    BUILD_TRUTH = False

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
    def cleanup():

        for filename in os.listdir(Parameters.output_path):
            os.remove(os.path.join(Parameters.output_path, filename))

    @staticmethod
    @pytest.fixture()
    def load_dataset():

        digits = load_digits()
        return digits.data

    @pytest.mark.parametrize("PBC,size,init,\
                              metric,topology,neighborhood_fun,\
                              train_algo,epochs,early_stop,\
                              clustering,plotall,load", [
        (False, 10, 'random', 'euclidean', 'hexagonal',
         'gaussian', 'online', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal',
         'gaussian', 'online', 10, None, None, False, True),
        (False, 10, 'PCA', 'cosine', 'hexagonal', 'gaussian',
         'online', 10, None, None, False, False),
        (False, 10, 'PCA', 'manhattan', 'hexagonal',
         'gaussian', 'online', 10, None, None, False, False),
        (True, 10, 'PCA', 'euclidean', 'hexagonal',
         'gaussian', 'online', 10, None, None, False, False),
        (True, 10, 'PCA', 'euclidean', 'square',
         'gaussian', 'online', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal',
         'gaussian', 'batch', 10, None, None, False, False),
        (False, 10, 'PCA', 'cosine', 'hexagonal',
         'gaussian', 'batch', 10, None, None, False, False),
        (False, 10, 'PCA', 'manhattan', 'hexagonal',
         'gaussian', 'batch', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal',
         'mexican_hat', 'batch', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal',
         'bubble', 'batch', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian',
         'batch', 100, 'mapdiff', None, False, False),
        (False, 10, 'PCA', 'euclidean', 'hexagonal', 'gaussian',
         'online', 100, 'mapdiff', None, False, False),
        (False, 10, 'PCA', 'euclidean', 'square', 'gaussian',
         'online', 10, None, None, False, False),
        (False, 10, 'PCA', 'euclidean', 'square', 'gaussian', 'batch',
         10, 'mapdiff', 'AgglomerativeClustering', True, False),
        (False, 10, 'PCA', 'euclidean', 'square', 'gaussian',
         'batch', 10, 'mapdiff', 'DBSCAN', True, False),
        (False, 10, 'PCA', 'euclidean', 'square', 'gaussian',
         'batch', 10, 'mapdiff', 'KMeans', True, False),
        (True, 10, 'PCA', 'euclidean', 'hexagonal',
         'gaussian', 'batch', 10, None, None, False, False),
        (True, 10, 'PCA', 'euclidean', 'square',
         'gaussian', 'batch', 10, None, None, False, False)
    ])
    @pytest.mark.parametrize("GPU", Parameters.GPU)
    def test_som(self, load_dataset, PBC, size, init,
                 metric, topology, neighborhood_fun,
                 train_algo, epochs, early_stop,
                 clustering, plotall, load, GPU):

        data = load_dataset
        system = platform.system()
        system = 'Darwin' if system == 'Windows' else system
        hashed_name = base64.urlsafe_b64encode(hashlib.md5(''.join([str(PBC), str(init),
                           str(metric), str(topology), str(neighborhood_fun),
                           str(train_algo), str(epochs), str(early_stop),
                           str(clustering), system]).encode('utf-8')).digest()).decode('ascii')

        net = sps.SOMNet(size, size, data,
                         topology=topology, PBC=PBC,
                         init=init, metric=metric,
                         random_seed=32, GPU=GPU, debug=False,
                         output_path=Parameters.output_path)
        net.train(train_algo=train_algo, start_learning_rate=0.01, epochs=epochs,
                  batch_size=-1, early_stop=early_stop)

        net.save_map("trained_som_{}.npy".format(hashed_name))
        assert (os.path.isfile(os.path.join(Parameters.output_path,
                                            "trained_som_{}.npy".format(hashed_name))))

        if self.BUILD_TRUTH:
            shutil.copyfile(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                            os.path.join(Parameters.truth_path, "trained_som_{}.npy".format(hashed_name)))
        
        decimal = 4

        print(np.max(np.load(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                             allow_pickle=True) - np.load(
            os.path.join(Parameters.truth_path, "trained_som_{}.npy".format(hashed_name)), allow_pickle=True)))
        assert_array_almost_equal(
            np.load(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                    allow_pickle=True),
            np.load(os.path.join(Parameters.truth_path, "trained_som_{}.npy".format(
                hashed_name)), allow_pickle=True),
            decimal=decimal)

        if load:
            net_l = sps.SOMNet(size, size, data,
                               load_file=os.path.join(
                                   Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                               topology=topology, PBC=PBC,
                               init=init, metric=metric,
                               random_seed=32, GPU=GPU, debug=False,
                               output_path=Parameters.output_path)

            weights_array = [[float(net_l.height)] * net_l.nodes_list[0].weights.shape[0],
                             [float(net_l.width)] *
                             net_l.nodes_list[0].weights.shape[0],
                             [float(net_l.PBC)] * net_l.nodes_list[0].weights.shape[0]] + \
                [net_l._get(node.weights) for node in net_l.nodes_list]
            assert_array_equal(np.array(weights_array),
                               np.load(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                                       allow_pickle=True))

        if plotall:
            net.plot_map_by_feature(feature_ix=1, show=False, print_out=True)
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_feature_1.png")))

            net.plot_map_by_difference(
                show=False, print_out=True, returns=False)
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_difference.png")))

            net.plot_projected_points(net.project_onto_map(data),
                                      project=False, jitter=True,
                                      show=False, print_out=True)
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_projected.png")))

            net.plot_projected_points(data, color_val=[n.difference for n in net.nodes_list],
                                      project=True, jitter=True,
                                      show=False, print_out=True,
                                      file_name=os.path.join(Parameters.output_path, "som_projected_2.png"))
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_projected_2.png")))

            clus_kwargs = {}
            if clustering == 'KMeans':
                clus_kwargs['random_state'] = 32

            labs, _ = net.cluster(
                data, algorithm=clustering, file_name="som_clusters_{}.npy".format(hashed_name), **clus_kwargs)
            assert (os.path.isfile(os.path.join(Parameters.output_path,
                                                "som_clusters_{}.npy".format(hashed_name))))
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_projected_" + clustering + ".npy")))

            if self.BUILD_TRUTH:
                shutil.copyfile(os.path.join(Parameters.output_path, "som_clusters_{}.npy".format(hashed_name)),
                                os.path.join(Parameters.truth_path, "som_clusters_{}.npy".format(hashed_name)))

                shutil.copyfile(os.path.join(Parameters.output_path, "som_projected_" + clustering + ".npy"),
                                os.path.join(Parameters.truth_path, "som_projected_{}.npy".format(hashed_name)))

            assert_array_almost_equal(
                np.load(os.path.join(Parameters.output_path, "som_clusters_{}.npy".format(hashed_name)),
                        allow_pickle=True),
                np.load(os.path.join(Parameters.truth_path, "som_clusters_{}.npy".format(hashed_name)),
                        allow_pickle=True), decimal=4)

            assert_array_almost_equal(
                np.load(os.path.join(Parameters.output_path, "som_projected_" + clustering + ".npy"),
                        allow_pickle=True),
                np.load(os.path.join(Parameters.truth_path, "som_projected_{}.npy".format(hashed_name)),
                        allow_pickle=True), decimal=4)

            net.plot_clusters(data, labs, project=True, show=False, print_out=True,
                              file_name=os.path.join(Parameters.output_path, "som_clusters.png"))
            assert (os.path.isfile(os.path.join(
                Parameters.output_path, "som_clusters.png")))

            if early_stop is not None:
                net.plot_convergence(show=False, print_out=True,
                                     file_name=os.path.join(Parameters.output_path, "som_convergence.png"))
                assert (os.path.isfile(os.path.join(
                    Parameters.output_path, "som_convergence.png")))

        self.cleanup()


if __name__ == '__main__':
    pass
