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

    system = platform.system()
    system = 'Darwin' if system == 'Windows' else system

    GPU = [False]
    try:
        import cupy
        GPU += [True]
    except ImportError:
        pass
        print('No CuPy found, GPU tests will be skipped.')


class TestNetwork:
    BUILD_TRUTH = False

    @classmethod
    def setup_method(cls, capsys):

        if os.path.exists(Parameters.output_path):
            shutil.rmtree(Parameters.output_path)
        os.mkdir(Parameters.output_path)

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

    @pytest.mark.parametrize("PBC", [False, True])
    @pytest.mark.parametrize("init", ["random", "PCA"])
    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
    @pytest.mark.parametrize("topology", ["hexagonal", "square"])
    @pytest.mark.parametrize("neighborhood_fun", ["gaussian", "mexican_hat", "bubble"])
    @pytest.mark.parametrize("train_algo", ["online", "batch"])
    @pytest.mark.parametrize("GPU", Parameters.GPU)
    def test_som(self, load_dataset, PBC, init,
                 metric, topology, neighborhood_fun,
                 train_algo, GPU):

        size = 10
        epochs = 10

        data = load_dataset
        hashed_name = base64.urlsafe_b64encode(hashlib.md5(''.join([str(PBC), str(init),
                                                                    str(metric), str(topology), str(
                                                                        neighborhood_fun),
                                                                    str(train_algo), Parameters.system]).encode('utf-8')).digest()).decode('ascii')

        net = sps.SOMNet(size, size, data,
                         topology=topology, PBC=PBC,
                         init=init, metric=metric,
                         random_seed=32, GPU=GPU, debug=False,
                         output_path=Parameters.output_path)
        net.train(train_algo=train_algo, start_learning_rate=0.01, epochs=epochs,
                  batch_size=-1, early_stop=None)

        net.save_map("trained_som_{}.npy".format(hashed_name))
        assert (os.path.isfile(os.path.join(Parameters.output_path,
                                            "trained_som_{}.npy".format(hashed_name))))

        if self.BUILD_TRUTH:
            shutil.copyfile(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                            os.path.join(Parameters.truth_path, "trained_som_{}.npy".format(hashed_name)))

        decimal = 4

        assert_array_almost_equal(
            np.load(os.path.join(Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                    allow_pickle=True),
            np.load(os.path.join(Parameters.truth_path, "trained_som_{}.npy".format(
                hashed_name)), allow_pickle=True),
            decimal=decimal)

        self.cleanup()

    @pytest.mark.parametrize("GPU", Parameters.GPU)
    def test_load(self, load_dataset, GPU):

        data = load_dataset
        hashed_name = base64.urlsafe_b64encode(hashlib.md5(''.join([
            Parameters.system]).encode('utf-8')).digest()).decode('ascii')
        
        net = sps.SOMNet(10, 10, data,
                         topology='hexagonal', PBC=False,
                         init='random', metric='euclidean',
                         random_seed=32, GPU=GPU, debug=False,
                         output_path=Parameters.output_path)
        net.train(train_algo='batch', start_learning_rate=0.01, epochs=10,
                  batch_size=-1, early_stop=None)
        net.save_map("trained_som_{}.npy".format(hashed_name))

        net_l = sps.SOMNet(10, 10, data,
                           load_file=os.path.join(
                               Parameters.output_path, "trained_som_{}.npy".format(hashed_name)),
                           topology='hexagonal', PBC=False,
                           init='random', metric='euclidean',
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

        self.cleanup()

    @pytest.mark.parametrize("clustering", ["AgglomerativeClustering", "DBSCAN", 'KMeans'])
    @pytest.mark.parametrize("GPU", Parameters.GPU)
    def test_clustering(self, load_dataset, clustering, GPU):

        data = load_dataset
        hashed_name = base64.urlsafe_b64encode(hashlib.md5(''.join([
            str(clustering), Parameters.system]).encode('utf-8')).digest()).decode('ascii')

        net = sps.SOMNet(10, 10, data,
                         topology='hexagonal', PBC=False,
                         init='random', metric='euclidean',
                         random_seed=32, GPU=GPU, debug=False,
                         output_path=Parameters.output_path)
        net.train(train_algo='batch', start_learning_rate=0.01, epochs=10,
                  batch_size=-1, early_stop=None)

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

        self.cleanup()


if __name__ == '__main__':
    pass
