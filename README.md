# SimpSOM (Simple Self Organizing Maps) 
## Version 1.0.0

SimpSOM is a basic implementation of Kohonen Self Organising Maps (SOM) for Python 2.7, 
useful for unsupervised learning, clustering and dimensionality reduction.

It allows to build and train SOM on your dataset, save/load the trained network weights, and display or print graphs 
of the network with selected features. 
If run as a standalone program, simpsom.py will run a toy model, where a number of colors will be mapped from the 3D
RGB space to the 2D network map and clustered according to their similarity in the origin space.



## Example of usage:

```
#Build a network 20x20 with a weights format taken from the raw_data. 
net = somNet(20, 20, raw_data)

#Train the network for 10000 epochs and with initial learning rate of 0.1. 
net.train(10000, 0.01)

#Save the weights to file
net.save('colorExample_weights')
	
#Print a map of the network nodes and colour them according to the first feature of the dataset
	and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()
	
#Project the datapoints on the new 2D network map.
net.proj_graph(raw_data, labels=labels)
```

## TO DOs:

- Implement a way to extract the obtained clusters as indexes.