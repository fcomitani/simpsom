"""
SimpSOM (Simple Self-Organizing Maps) v1.3.5
F. Comitani @2017-2021 
 
A lightweight python library for Kohonen Self-Organizing Maps (SOM).
"""

from __future__ import print_function

import sys
import os, errno
import numpy as np

from simpsom.network import SOMNet
from simpsom._version import __version__

def run_colors_example(train_algo='batch', epochs=-1, early_stop=None, out_path='./', GPU=False):   

    """Example of usage of SimpSOM: a number of vectors of length three
        (corresponding to the RGB values of a color) are used to briefly train a small network.
        Different example graphs are then printed from the trained network.     

        Args:
	    train_algo (str): training algorithm, choose between 'online' or 'batch' 
                (default 'batch').
        epochs (int): Number of training iterations. If not selected (or -1)
            automatically set epochs as 10 times the number of datapoints.
        early_stop (str): Early stopping method, for now only 'mapdiff' (checks if the
            weights of nodes don't change) and 'bmudiff' (checks if the assigned bmu to each sample
            don't change) are available. If None, don't use early stopping (default None).
        out_path (str, optional): path to the output folder.
    """ 

    """ Set up output folder. """

    if out_path != './':
        try:
            os.makedirs(out_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
                
    raw_data = np.asarray([[1, 0, 0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.2,0.2,0.5]])
    labels   = ['red','green','blue','yellow','magenta','cyan','indigo']

    print("Welcome to SimpSOM (Simple Self Organizing Maps) v"+__version__+"!\nHere is a quick example of what this library can do.\n")
    print("The algorithm will now try to map the following colors: ", end=" ")
    for i in range(len(labels)-1):
            print((labels[i] + ", "), end=" ") 
    print("and " + labels[-1]+ ".\n")
    
    net = SOMNet(20, 20, raw_data, PBC=True, init='random', GPU=GPU) 
    
    net.color_ex = True
    net.train(train_algo=train_algo, start_learning_rate=0.01, epochs=epochs,
                early_stop=early_stop, early_stop_patience=3, early_stop_tolerance=1e-4)

    print("Saving weights and a few graphs...", end=" ")
    net.save('color_example_weights', out_path=out_path)
    net.nodes_graph(out_path=out_path)
    
    net.diff_graph(out_path=out_path)
    test = net.project(raw_data, labels=labels, out_path=out_path)
    net.cluster(raw_data, clus_type='qthresh', out_path=out_path) 
    
    print("done!")

if __name__ == "__main__":

    run_colors_example()
