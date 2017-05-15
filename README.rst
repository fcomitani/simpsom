SimpSOM (Simple Self Organizing Maps) 
=====================================

Version 1.1.1
-------------

SimpSOM is a lightweight implementation of Kohonen Self Organising Maps (SOM) for Python 2.7, 
useful for unsupervised learning, clustering and dimensionality reduction.

It allows to build and train SOM on your dataset, save/load the trained network weights, and display or print graphs 
of the network with selected features. 
The function ``run_colorsExample()`` will run a toy model, where a number of colors will be mapped from the 3D
RGB space to the 2D network map and clustered according to their similarity in the origin space.

Dependencies
------------

- Numpy 1.11.0 (older versions may work);
- Matplotlib 1.5.1 (older versions may work);
- Sklearn 0.15 (older versions may work), optional, needed only for clustering with algorithms other 
	than Quality Threshold. (use the option ``-e .[cluster]`` when installing)

Example of Usage:
-----------------


	#Import the library
	import SimpSOM as sps

	#Build a network 20x20 with a weights format taken from the raw_data. 
	net = sps.somNet(20, 20, raw_data)

	#Train the network for 10000 epochs and with initial learning rate of 0.1. 
	net.train(10000, 0.01)

	#Save the weights to file
	net.save('colorExample_weights')
		
	#Print a map of the network nodes and colour them according to the first feature of the dataset
	#and then according to the distance between each node and its neighbours.
	net.nodes_graph(colnum=0)
	net.diff_graph()
		
	#Project the datapoints on the new 2D network map.
	net.project(raw_data, labels=labels)

	#Cluster the datapoints according to the Mean Shift algorithm from sklearn.
	net.cluster(raw_data, type='MeanShift')

	
Version 1.1.1 What's New
------------------------

- Clustering is now possible with the ``cluster`` command,
	Quality Threshold and a few sklearn clustering algorithms are availabe;
- It is now possible to install SimpSOM through ``python setup.py install``.
	
TO DOs:
-------

- Update the available cluster algorithms from sklearn;
- Add compatibility with cvs format.