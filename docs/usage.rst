.. _usage:

===========
Basic Usage
===========

Map Instantiation
=================


Using `simpsom` is easy, after loading the library initialize a network object.

.. code-block:: python

    import simpsom as sps

    net = sps.SOMNet(20, 20, data, topology='hexagonal', 
                    init='PCA', metric='cosine',
                    neighborhood_fun='gaussian', PBC=True,
                    random_seed=32, GPU=False, CUML=False,
                    output_path="./")


First, you need to provide `length` and `height` of your network, in this case we 
will build a 20x20 grid. Your input `data` matrix must be in a `np.ndarray`-compatible format.

Two tiling topologies are available in `simpsom`, a `square` grid or and `hexagonal` grid.
The network nodes will be initialized either randomly with `init='random'` (the default choice),
or by sampling from the space spanned by the first two principal components with `'PCA'`. 
Since the initialization is stochastic the `random_seed` can be set.

`metric` expects a distance metric to evaluate the closest matching unit during training or inference. 
Make sure to select a metric that is appropriate for your starting dimensionality and dataset.
`neighborhood_fun` allows you to choose a neighborhood selection function for training. 
Currently, `'gaussian'`, `'mexican_hat'` and `'bubble'` are available.

Periodic Boundary Conditions can be activated with the `PBC` flag .

Finally, `GPU` and `CUML` can be activated if `CuPy` and `CuML` are available on your system
to run training and clustering on GPU respectively.

Files will be saved in the provided `output_path`

See the :ref:`api` for details on these and more available options.

Training
========

The network can be trained with the `train` function.
Two `train_algo` training algorithms are available: `'online'` and `'batch'`.

.. code-block:: python

    net.train(train_algo='batch', start_learning_rate=0.01, epochs=-1, 
        batch_size=-1)

The starting learning rate (`start_learning_rate`), and the number of `epochs`
can be customized. 
*Note:* Beware that by design, one epoch in `'online'` training corresponds to a single point
being looked at to the map, while in `'batch'` training, the whole dataset will be seen at each epoch.
`simpsom` will automatically select the best number of epochs
if the `epochs` parameter is not provided or set to `-1`. 10x the number of samples
in `'online'` training and 10 epochs in `'batch'` training.

Automatically stopping the training at convergence is possible with the `early_stop` flag, 
but this functionality is still a work in progress. Activate it only if you are sure of what you are doing.

Once the map has been trained, you can save its weights with `save`.

.. code-block:: python

    net.save_map("./trained_som.npy")

A saved map file can be then loaded into a new `net` object by providing its path with the 
`load_file` flag during initialization.

Analyzing the Results
=====================

Node weights can be accessed by the `weights` attribute from each `SOMNode` object, their position
with `pos`. The nodes can be listed with `node_list`.
To recover the positions and weights of all nodes for example

.. code-block:: python

    all_positions = [[node.pos[0], node.pos[1]] for node in self.nodes_list]
    all_weights = [node.weights[feature] for node in self.nodes_list]

The `project_onto_map` method allows you to project data points onto a 2D-trained map.

.. code-block:: python

    projected_data = net.project_onto_map(data)

Plotting
========

`simpsom` comes with several plotting functionalities that allow you to inspect
the results of the training.

.. code-block:: python

    net.plot_map_by_difference(show=True, print_out=True)


`plot_map_by_difference` allows you to visualize the trained map coloring each node
by the average weights difference from its neighbors. Useful to identify map boundaries.
The map can be saved to a file by providing a path and filename to `print_out`,
while the `show` flag is needed for interactive visualization.

`plot_map_by_difference` will automatically calculate the nodes' differences and save them
to the `difference` attribute within each `SOMNode` object by calling the `net.self.get_nodes_difference()`
method under the hood.

The map can also be plotted by coloring nodes by a specific feature, by providing its 
index with `plot_map_by_feature`.

.. code-block:: python

    _ = net.plot_map_by_feature(feature=100, show=True, print_out=True)


Datapoints (old or new) can be projected onto the map and visualized as a scatterplot
with the `project_onto_map` and `plot_projecte_points` functions.

.. code-block:: python

    projected_data = net.project_onto_map(data)
    net.plot_projected_points(projected_data, color_val=[n.difference for n in net.nodes_list],
            project=False, jitter=False, 
            show=True, print_out=False)

The points can be projected on the fly without having to run `project_onto_map` before plotting 
by activating the `project` flag.
`color_val` allows you to pass color values for the nodes in case you want to compare the projection
with a specific weight value from the nodes. 
`jitter` lets you add random jitter to avoid overlapping points.

If early stopping was activated (not recommended) a convergence plot can be
produced with `plot_convergence`.

.. code-block:: python

    net.plot_convergence(fsize=(5, 5), logax=False)


Clustering
==========

Clustering is one of the many uses of SOM.
Once the points are projected onto the 2-dimensional SOM, they can be used with any clustering algorithm.
`simpsom` provides a wrapper function to directly run a clustering identification analysis
on the projected points.

.. code-block:: python

    labs, points = net.cluster(data, algorithm='KMeans', n_clusters=10)

Any `scikit-learn` or `CuML` algorithm can be employed, the name must be passed with `algorithm`
and any further argument required by the algorithm (e.g. `n_clusters` for `'KMeans'`) can also be passed.
By default the function will assume the data has been provided in its original dimensionality, but pre-projected
data points can be also provided by deactivating the `project` flag.

*Note:* for clustering algorithms that require a distance metric to be passed, it is recommended you make sure
to provide a PBC-compatible metric if PBC were activated during training.

For example, this can be done through `partial` and by exploiting the pbc-compatible distance wrapper
provided with the chosen topology (`net.polygons.distance_pbc`) as in the following example.

.. code-block:: python

    labs, points = net.cluster(data, algorithm='DBSCAN', metric=partial(net.polygons.distance_pbc,
                                                                net_shape=(net.width, net.height),
                                                                distance_func=lambda x, y: net.xp.sqrt(net.xp.sum(net.xp.square(x-y))),
                                                                xp=net.xp), eps=.1)



