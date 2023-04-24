import sys
sys.path.append("/simpsom/")
import simpsom as sps 
# import simpsom as sps
import random
import numpy as np
random.seed(42)
np.random.seed(42)
# data = np.random.rand(10,2)
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]  
print(data)

net = sps.SOMNet(3, 3, data, topology='hexagonal', 
                init='pca', metric='euclidean',
                neighborhood_fun='gaussian', PBC=False,
                random_seed=32, GPU=False, CUML=False,
                output_path="./")

net.train(train_algo='online', start_learning_rate=0.01, epochs=5)

for node in net.nodes_list:
    print(node.weights)
    
    
    
net = sps.SOMNet(3, 3, data, topology='hexagonal', 
                init='random', metric='euclidean',
                neighborhood_fun='gaussian', PBC=False,
                random_seed=32, GPU=True, CUML=False,
                output_path="./")

net.train(train_algo='batch', start_learning_rate=0.01, epochs=5, batch_size=1)

for node in net.nodes_list:
    print(node.weights)