# SimpSOM (Simple Self-Organizing Maps)

[![DOI](https://zenodo.org/badge/91130860.svg)](https://zenodo.org/badge/latestdoi/91130860)
[![PyPI version](https://badge.fury.io/py/SimpSOM.svg)](https://badge.fury.io/py/simpsom)
[![Build Status](https://travis-ci.org/fcomitani/simpsom.svg?branch=master)](https://travis-ci.org/fcomitani/simpsom)
[![Documentation Status](https://readthedocs.org/projects/simpsom/badge/?version=latest)](https://simpsom.readthedocs.io/en/latest/?badge=latest)

The version contained in this branch is currently under development.
Please use the main branch if you are looking for a stable version (1.3.4).

## Version 2.0.0

SimpSOM is a lightweight implementation of Kohonen Self-Organizing Maps
(SOM) for Python 3, useful for unsupervised learning,
clustering and dimensionality reduction.

The package is now available on PyPI, to retrieve it just type
`pip install SimpSOM` or download it from here and install with
`python setup.py install`.

It allows you to build and train SOM on your dataset, save/load the trained
network weights, and display or print graphs of the network with
selected features. The function `run_colors_example()` will run a toy
model, where a number of colors will be mapped from the 3D RGB space to
the 2D network map and clustered according to their similarity in the
origin space.

![](./docs/figs/colorExample.png)

## What\'s New

- Class and function names have been updated to adhere to PEP8.
- Batch training has been added and is now the default algorithm.
- A light parallelization is nor possible with RAPIDS.

## Version compatibility

This version introduces a number of changes, while attempting to maintain
the original philosophy of this project: a SOM library easy to understand and customize.
Functions and classes names have been changed to improve readability.
If you are migrating from an older version (<=1.3.4), please make sure to check the API first!

## Dependencies

```
-   Numpy		== 1.19.5 
-   Matplotlib	== 3.3.3 
-   Sklearn		== 0.22.2.post1 
```

Older/newer versons may work, but make sure to test the library


## Example of Usage

Here is a quick example on how to use the library with an exemplary `raw_data`
dataset:

    #Import the library
    import simpsom as sps

    #Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions. 
    net = sps.SOMNet(20, 20, raw_data, PBC=True)

    #By default the network will be trained with the batch training algorithm and 10xsamples number of epochs.
    #No learning rate is needed.
    net.train()

    #Alternatively, all of these options can be set mantually. 
    #For example to train the network with online training (much slower!)
    #for 1000 epochs and with initial learning rate of 0.01, use:
    #net.train(train_algo='online', learning_rat=0.01, epochs=1000)

    #Save the weights to file
    net.save('filename_weights')

    #Information on each node is stored in the .nodeList attribute of the network. These include each node position
    #in the hexagonal grid (.pos) or its weights (.weights), i.e. the position of the node in the features space.
    position_node0 = net.node_list[0].pos
    weights_node0 = net.node_list_[0].weights 

    #Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
    #and then according to the distance between each node and its neighbours.
    net.nodes_graph(colnum=0)
    net.diff_graph()

    #Project the datapoints on the new 2D network map.
    net.project(raw_data, labels=labels)

    #Cluster the datapoints according to the Quality Threshold algorithm.
    #It's important to note that only Quality Threshold ('qthrehs') and Density Peak 'dpeak'
    #are compatible with periodic boundary conditions. Deactivate PBC if you intend to use
    #'MeanShift', 'DBSCAN', 'KMeans', or your own clustering tool.
    net.cluster(raw_data, clus_type='qthresh')	
    
 ## Running on GPU
 
 If you have a CUDA compatible system, you can run this library on GPU just by activating the
 namesake flag when instatiating the SOMNet class.
 
 	net = sps.SOMNet(20, 20, raw_data, PBC=False, GPU=True)
	
To be able to run this option you will need the following RAPIDS libraries:

```
- cupy == 8.60
- cuml == 0.18
```

Please note that newer versions may work, but this libraries are still in development and API breaking
changes are to be expected, make sure to check the RAPIDS repositories for more information.
These libraries do not need to be installed to run the CPU version of SimpSOM.
	
## A More Interesting Example: MNIST

Here is another example of SimpSOM capabilites: the library was used to try and reduce a MNIST handwritten digits dataset. A 50x50 nodes map was trained with 500 MINST landmark datapoints and 100000 epochs in total, starting from a 0.1 learning rate and without PCA Initialization.

![](./docs/figs/nD_annotated.png)

Projecting a few of those points on the map gives the following result, showing a clear distinction between cluster of digits with a few exceptions. Similar shapes (such as 7 and 9) are mapped closed together, while relatively far from other more distinct digits. The accuracy of this mapping could be further improved by tweaking the map parameters and training.
	
## Documentation

See [here](https://simpsom.readthedocs.io/en/master/) the full API documentation

## Citation

If using this library, please cite it as

> Federico Comitani, 2019. fcomitani/SimpSOM: v1.3.4. doi:10.5281/zenodo.2621560

