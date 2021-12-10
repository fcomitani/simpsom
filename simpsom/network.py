"""
SimpSOM (Simple Self-Organizing Maps) v2.0.0
F. Comitani @2017-2021 
 
A lightweight python library for Kohonen Self-Organizing Maps (SOM).
"""

from __future__ import print_function

import sys
import os, errno

import random 
from math import sqrt, exp, log

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import simpsom.interface as interface
import simpsom.hexagons as hx
from simpsom.cluster import density_peak as dp
from simpsom.cluster import quality_threshold as qt

class SOMNet:
    """ Kohonen SOM Network class. """

    def __init__(self, net_height, net_width, data, load_file=None, metric='euclidean', metric_kwds={},
                 init='PCA', PBC=False, GPU=False, random_seed=None):

        """Initialise the SOM network.

        Args:
            net_height (int): Number of nodes along the first dimension.
            net_width (int): Numer of nodes along the second dimension.
            data (self.interface.num.array or list): N-dimensional dataset.
            load_file (str, optional): Name of file to load containing information 
                to initialize the network weights.
            metric (string): distance metric for the identification of best matching
                units. Accepts metrics available in scikit-learn (default 'euclidean').
            metric_kwds (dict): dictionary with optional keywords to pass to the chosen
                metric (default {}).
            init (str or list of self.interface.num.array): Nodes initialization method, to be chosen between 'random'
                or 'PCA' (default 'PCA'). Alternatively a couple of vectors can be provided
                whose values will be spanned uniformly.
            PBC (boolean): Activate/deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC (default 0).
            GPU (boolean): Activate/deactivate GPU run with RAPIDS (requires CUDA).
            random_seed (int): Seed for the random numbers generator (default None).
                
        """

        if random_seed is not None:
            random.seed(random_seed)

        """ Set CPU/GPU libraries. """

        self.interface = interface.InterfaceGPU() if bool(GPU)\
                            else interface.InterfaceCPU() 

        """ Switch to activate special workflow if running the colors example. """
        self.color_ex = False

        """ Switch to activate periodic boundary conditions. """
        self.PBC = bool(PBC)

        if self.PBC:
            print("Periodic Boundary Conditions active.")
        else:
            print("Periodic Boundary Conditions inactive.")

        self.node_list = []
        self.data = self.interface.num.array(data)\
                         .astype(self.interface.num.float32)
        
        self.metric      = metric
        self.metric_kwds = metric_kwds
        
        self.convergence   = []

        """ Load the weights from file, generate them randomly or from PCA. """

        init_vec     = None
        init_bounds  = None
        wei_array    = None

        if load_file is None:
            self.net_height = net_height
            self.net_width  = net_width

            if init == 'PCA':
                print("The weights will be initialized with PCA.")
            
                pca     = self.interface.PCA(n_components = 2)
                pca.fit(self.data)
                init_vec = pca.components_
            
            elif init == 'random':
                print("The weights will be initialized randomly.")

                for i in range(self.data.shape[1]):
                    init_bounds = [self.interface.num.min(self.data,axis=0),
                                   self.interface.num.max(self.data,axis=0)]
            
            else: 
                print("Custom weights provided.")

                init_vec = init

        else:   
            print('The weights will be loaded from file.')

            if load_file.endswith('.npy')==False:
                load_file = load_file+'.npy'
            wei_array = self.interface.num.load(load_file)
            #add something to check that data and array have the same dimensions,
            #or that they are mutually exclusive
            self.net_height = int(wei_array[0][0])
            self.net_width  = int(wei_array[0][1])
            self.PBC        = bool(wei_array[0][2])

        """ When loaded from file, element 0 contains information on the network shape."""
        count_wei = 1

        """ Set the weights. """

        for x in range(self.net_width):
            for y in range(self.net_height):

                #if weights were loaded from file
                this_wei = wei_array[count_wei] if wei_array is not None\
                                                else None
                count_wei += 1

                self.node_list.append(SOMNode(x, y, self.data.shape[1], \
                    self.net_height, self.net_width, self.PBC, self.interface,\
                    wei_bounds=init_bounds, init_vec=init_vec, wei_array=this_wei))

    def save(self, fileName='SOMNet_trained', out_path='./'):
    
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            fileName (str, optional): Name of file where the data will be saved.
            out_path (str, optional): Path to the folder where data will be saved.
        """
        
        wei_array = [self.interface.num.zeros(len(self.node_list[0].weights))]
        wei_array[0][0], wei_array[0][1], wei_array[0][2] = self.net_height, self.net_width, int(self.PBC)
        for node in self.node_list:
            wei_array.append(node.weights)
        self.interface.num.save(os.path.join(out_path,fileName), self.interface.num.asarray(wei_array))
    

    def update_sigma(self, n_iter):
    
        """Update the gaussian sigma.

        Args:           
            n_iter (int): Iteration number.
            
        """
    
        self.sigma = self.start_sigma * exp(-n_iter/self.tau)
    

    def update_learning_rate(self, n_iter):
    
        """Update the learning rate.

        Args:           
            n_iter (int): Iteration number.
            
        """
        
        self.learning_rate =  self.start_learning_rate * exp(n_iter/self.epochs)
    

    def find_bmu_ix(self, vecs):
    
        """Find the index of the best matching unit (BMU) for a given list of vectors.

        Args:           
            vec (2d self.interface.num.array or list of lists): vectors whose distance from the network
                nodes will be calculated.
            
        Returns:            
            bmu (SOMNode): The best matching unit node index.
            
        """
        
        dists = self.interface.pairdist(vecs,
                        self.interface.num.array([n.weights for n in self.node_list]), 
                        metric=self.metric, **self.metric_kwds)

        return self.interface.num.argmin(dists,axis=1)
   
    def train(self, train_algo='batch', epochs=-1, start_learning_rate=0.01,  
                early_stop=None, early_stop_patience=3, early_stop_tolerance=1e-4,
                batch_size=-1):
    
        """Train the SOM.

        Args:
            train_algo (str): training algorithm, choose between 'online' or 'batch' 
                (default 'online'). Beware that the online algorithm will run one datapoint
                per epoch, while the batch algorithm runs all points at one for each epoch.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints. 
            start_learning_rate (float): Initial learning rate, used only in online
                learning.
            early_stop (str): Early stopping method, for now only 'mapdiff' (checks if the
                weights of nodes don't change) and 'bmudiff' (checks if the assigned bmu to each sample
                don't change) are available. If None, don't use early stopping (default None).
            early_stop_patience (int): Number of iterations without improvement before stopping the 
                training, only available for batch training (default 3).
            early_stop_tolerance (float): Improvement tolerance, if the map does not improve beyond
                this threshold, the early stopping counter will be activated (it needs to be set
                appropriately depending on the used distance metric). Ignored if early stopping
                is off (default 1e-4).
            batch_size (int): Split the dataset in batches of this size when calculating the 
                new weights, works only when train_algo is 'batch' and helps keeping down the 
                memory requirements when working with large datasets, if -1 run the whole dataset
                at once. 

        """

        print("The map will be trained with the "+train_algo+" algorithm.")
        self.start_sigma = max(self.net_height, self.net_width)/2
        self.start_learning_rate = start_learning_rate

        if epochs == -1:
            epochs  = self.data.shape[0]*10
            
        self.epochs = epochs
        self.tau    = self.epochs/log(self.start_sigma)

        if batch_size == -1 or batch_size > self.data.shape[0]:
            batch_size = self.data.shape[0]

        if train_algo == 'online':
            """ Online training.
                Bootstrap: one datapoint is extracted randomly with replacement at each epoch 
                and used to update the weights.
            """

            for n_iter in range(self.epochs):

                if n_iter%10==0:
                    print("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)), end=' ')

                self.update_sigma(n_iter)
                self.update_learning_rate(n_iter)

                input_vec = self.data[random.randint(0, self.data.shape[0]-1), :].reshape(1,self.data.shape[1])
                
                bmu = self.node_list[int(self.find_bmu_ix(input_vec)[0])]

                for node in self.node_list:
                    node.update_weights(input_vec[0], self.sigma, self.learning_rate, bmu)

        elif train_algo == 'batch':
            """ Batch training.
                All datapoints are used at once for each epoch, 
                the weights are updated with the sum of contributions from all these points.
                No learning rate needed.

                Kinouchi, M. et al. "Quick Learning for Batch-Learning Self-Organizing Map" (2002).
            """
    
            #Storing the distances and weight matrices defeats the purpose of having
            #nodes as instances of a class, but it helps with the optimization
            #and parallelization at the cost of memory.
            #The object-oriented structure is kept to simplify code reading. 

            """ Calculate the square euclidean distance matrix to speed up batch training. """


            dist_matrix_sq = self.interface.num.zeros((self.net_width*self.net_height, 
                                        self.net_width*self.net_height)) 

            for i in range(self.net_width*self.net_height):
                for j in range(i + 1, self.net_width*self.net_height):
                    dist_matrix_sq[i,j] = self.interface.num.linalg.norm(self.node_list[i].pos-self.node_list[j].pos)
            
            dist_matrix_sq += dist_matrix_sq.T
            dist_matrix_sq *= dist_matrix_sq

            """ Store all weights as matrix. """

            all_weights = self.interface.num.array([n.weights for n in self.node_list])

            early_stop_counter = 0

            for n_iter in range(self.epochs):

                """ Early stop check. """

                if early_stop_counter == early_stop_patience:

                    print("\rTolerance reached at epoch {:d}, stopping training.".format(n_iter-1))

                    break

                self.update_sigma(n_iter)
                
                """ Matrix of gaussian effects. """

                gauss = self.interface.num.exp(-dist_matrix_sq/(2*self.sigma*self.sigma))

                if n_iter%10==0:
                    print("\rTraining SOM... {:d}%".format(int(n_iter*100.0/self.epochs)), end=' ')
                        
                """ Run through mini batches to ease the memory burden. """

                numerator   = self.interface.num.zeros((self.net_width*self.net_height, 
                                        self.data.shape[1])) 
                denominator =  self.interface.num.zeros((self.net_width*self.net_height, 
                                        self.data.shape[1]))

                for b in range(int((self.data.shape[0]+batch_size-1)/batch_size)):
                    
                    batchdata = self.data[b*batch_size : min((b+1)*batch_size,self.data.shape[0])]

                    """ Find BMUs for all points and subselect gaussian matrix. """

                    dists = self.interface.pairdist(batchdata, all_weights,  
                                    metric=self.metric, **self.metric_kwds)
                    bmus = self.interface.num.argmin(dists, axis=1)

                    batchgauss = gauss[bmus]
                    denominator += self.interface.num.repeat(batchgauss.sum(axis=0)[:, self.interface.num.newaxis], self.data.shape[1], axis=1)

                    batchgauss = self.interface.num.repeat(batchgauss[:, :, self.interface.num.newaxis], self.data.shape[1], axis=2)
                    batchdata = self.interface.num.repeat(batchdata[:, self.interface.num.newaxis, :], gauss.shape[1], axis=1)

                    numerator  += self.interface.num.multiply(batchgauss,batchdata).sum(axis=0)

                new_weights = self.interface.num.divide(numerator,denominator)
                new_weights[self.interface.num.isnan(new_weights)] = all_weights[self.interface.num.isnan(new_weights)] 

                """ Early stopping, active if patience is not None """

                if early_stop is not None:

                    #These are pretty ruough convergence tests
                    #To do: add more

                    if early_stop == 'mapdiff':      

                        """ Checks if the map weights are not moving. """

                        self.convergence.append(self.interface.pairdist(new_weights,all_weights,  
                                metric=self.metric, **self.metric_kwds).mean())
                    
                    elif early_stop == 'bmudiff':

                        """ Checks if the bmus mean distance from the samples has stopped improving. """

                        self.convergence.append(self.interface.num.min(dists,axis=1).mean())

                    else:
                        sys.exit('Error: convergence method not recognized. Choose between \'mapdiff\' and \'bmudiff\'.')

                    if n_iter > 0 and self.convergence[-2]-self.convergence[-1] < early_stop_tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter  = 0

                all_weights = new_weights

            """ Store final weights in the nodes objects. """
            #Revert back to object oriented
            
            for n_iter, node in enumerate(self.node_list):
                node.weights = all_weights[n_iter] # * self.learning_rate

            if early_stop is not None:

                """ Plot convergence if it was tracked. """
                
                self.plot_convergence(logax=False)

        else:
            sys.exit('Error: training algorithm not recognized. Choose between \'online\' and \'batch\'.')
                        
        print("\rTraining SOM... done!")

    def plot_convergence(self, logax=False, show=False, print_out=False, out_path='./'):

        """ Plot the the map training progress according to the 
            chosen convergence criterion.
            
        Args: 
            logax (bool, optional): if True, plot convergence on logarithmic y axis.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
        """


        fig=plt.figure(figsize=(30,10))
        fig, ax = plt.subplots()
        
        ax.set_facecolor('white')
        plt.grid(color='#aaaaaa')
        
        plt.plot(self.convergence, color='#444444')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        if logax:
            ax.set_yscale('log')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xlabel('Iteration', fontsize=15)
        plt.ylabel('Convergence score', fontsize=15)

        print_name = os.path.join(out_path,'convergence.png')
        
        if print_out == True:
            plt.savefig(print_name, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        if show != False and print_out != False:
            plt.clf()


    def nodes_graph(self, colnum=0, show=False, print_out=True, out_path='./', colname=None):
    
        """Plot a 2D map with hexagonal nodes and weights values

        Args:
            colnum (int): The index of the weight that will be shown as colormap.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.
            out_path (str, optional): Path to the folder where data will be saved.
        """

        if not colname:
            colname = str(colnum)

        centers = [[node.pos[0],node.pos[1]] for node in self.node_list]
        
        side    = sqrt(self.net_width*self.net_height)
        width_p = 100
        dpi     = 72
        xInch   = self.net_width*width_p/dpi 
        yInch   = self.net_height*width_p/dpi 
        fig     = plt.figure(figsize=(xInch, yInch), dpi=dpi)

        cols = [self.interface.get_value(node.weights) for node in self.node_list]
        
        if self.color_ex==True:
            ax = hx.plot_hex(fig, centers, cols, color_ex=self.color_ex)
            ax.set_title('Node Grid w Color Features', size=4*side)
            print_name=os.path.join(out_path,'nodes_colors.png')

        else:
            cols = [c[colnum] for c in cols]
            ax = hx.plot_hex(fig, centers, cols)
            ax.set_title('Node Grid w Feature ' +  colname, size=4*side)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            cbar=plt.colorbar(ax.collections[0], cax=cax)
            cbar.set_label(colname, size=4*side, labelpad=2.5*side)
            cbar.ax.tick_params(labelsize=2*side)
            cbar.outline.set_visible(False)

            plt.sca(ax)
            print_name=os.path.join(out_path,'nodes_feature_'+str(colnum)+'.png')
            
        if print_out==True:
            plt.savefig(print_name, bbox_inches='tight', dpi=dpi)
        if show==True:
            plt.show()
        if show!=False and printout!=False:
            plt.clf()

    def diff_graph(self, show=False, print_out=True, returns=False, out_path='./'):
    
        """Plot a 2D map with nodes and weights difference among neighboring nodes.

        Args:
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            returns (bool, optional): Choose to return the difference value.
            out_path (str, optional): Path to the folder where data will be saved.

        Returns:
            (list): difference value for each node.             
        """
        
        """ Find adjacent nodes in the grid. """
        
        #print(type(self.node_list[0].weights))
        #print(type(self.node_list[0].get_node_distance(self.node_list[1])))
        #print(self.node_list[0].get_node_distance(self.node_list[1])<= 1.001)
        #pippo=[[node2.weights for node2 in self.node_list \
        #                            if node != node2 and node.get_node_distance(node2) <= 1.001]
        #                             for node in self.node_list]
        #print(pippo, len(pippo), len(pippo[0]))
        #print(self.interface.num.array)
        #print(self.interface.num.array(pippo))
        

        neighbors = [self.interface.num.array([node2.weights for node2 in self.node_list \
                    if node != node2 and node.get_node_distance(node2) <= 1.001])
                    for node in self.node_list]

        """ Calculate the summed weight difference. """

        diffs = [self.interface.get_value(
                 self.interface.pairdist(n.weights.reshape(1,n.weights.shape[0]),
                                         neighbors[i], 
                                         metric='euclidean', **self.metric_kwds).mean())\
                 for i,n in enumerate(self.node_list)]

        """ Define plotting hexagon centers. """

        centers = [[node.pos[0],node.pos[1]] for node in self.node_list]

        """ Set up and plot. """

        if show == True or print_out==True:
        
            side    = sqrt(self.net_width*self.net_height)
            width_p = 100
            dpi     = 72
            x_inch  = self.net_width*width_p/dpi 
            y_inch  = self.net_height*width_p/dpi 
            fig     = plt.figure(figsize=(x_inch, y_inch), dpi=dpi)

            ax = hx.plot_hex(fig, centers, diffs)
            ax.set_title('Nodes Grid w Weights Difference', size=4*side)
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="5%", pad=0.0)
            cbar    = plt.colorbar(ax.collections[0], cax=cax)
            cbar.set_label('Weights Difference', size=4*side, labelpad=2.5*side)
            cbar.ax.tick_params(labelsize=2*side)
            cbar.outline.set_visible(False)

            plt.sca(ax)
            print_name = os.path.join(out_path,'nodes_difference.png')
            
            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight', dpi=dpi)
            if show == True:
                plt.show()
            if show != False and print_out != False:
                plt.clf()

        if returns == True:
            return diffs 

    def project(self, array, colnum=-1, labels=[], show=False, print_out=True, out_path='./', colname = None):

        """Project the datapoints of a given array to the 2D space of the 
            SOM by calculating the bmus. If requested plot a 2D map with as 
            implemented in nodes_graph and adds circles to the bmu
            of each datapoint in a given array.

        Args:
            array (self.interface.num.array): An array containing datapoints to be mapped.
            colnum (int): The index of the weight that will be shown as colormap. 
                If not chosen, the difference map will be used instead.
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
            colname (str, optional): Name of the column to be shown on the map.
            
        Returns:
            (list): bmu x,y position for each input array datapoint. 
            
        """
        
        if not colname:
            colname = str(colnum)

        if labels != []:
            colors  = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
                       '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
            counter = 0
            class_assignment = {}
            for i in range(len(labels)):
                if labels[i] not in class_assignment:
                    class_assignment[labels[i]] = colors[counter]
                    counter = (counter + 1)%len(colors)
        
        if not isinstance(array, self.interface.num.ndarray):
            array = self.interface.num.array(array)\
                    .astype(self.interface.num.float64)

        bmu_list, cls = [], []
        bmu_list = [self.node_list[int(mu)].pos for mu in self.find_bmu_ix(array)]
        
        if self.color_ex:
            cls = self.interface.get_value(array)
        else: 
            if labels != []:   
                cls = [class_assignment[labels[i]] for i in range(array.shape[0])]
            elif colnum == -1:
                cls = ['#ffffff']*array.shape[0]
            else: 
                cls = self.interface.get_value(array[:,colnum])

        if show == True or print_out == True:
        
            """ Call nodes_graph/diff_graph to first build the 2D map of the nodes. """
            
            side = sqrt(self.net_width*self.net_height)

            if self.color_ex == True:
                print_name = os.path.join(out_path,'color_projection.png')
                self.nodes_graph(colnum, False, False)
                plt.scatter([pos[0] for pos in bmu_list],[pos[1] for pos in bmu_list], color=cls,  
                        s=500, edgecolor='#ffffff', linewidth=5, zorder=10)
                plt.title('Datapoints Projection', size=4*side)
            else:
                #a random perturbation is added to the points positions so that data 
                #belonging plotted to the same bmu will be visible in the plot      
                if colnum == -1:
                    print_name = os.path.join(out_path,'projection_difference.png')
                    self.diff_graph(False, False, False)
                    plt.scatter([pos[0]-0.125+random.random()*0.25 for pos in bmu_list],
                                [pos[1]-0.125+random.random()*0.25 for pos in bmu_list], c=cls, cmap=cm.viridis,
                                 s=200, linewidth=0, zorder=10)
                    plt.title('Datapoints Projection on Nodes Difference', size=4*side)
                else:   
                    print_name = os.path.join(out_path,'projection_'+ colname +'.png')
                    self.nodes_graph(colnum, False, False, colname=colname)
                    plt.scatter([pos[0]-0.125+random.random()*0.25 for pos in bmu_list],
                                [pos[1]-0.125+random.random()*0.25 for pos in bmu_list], c=cls, cmap=cm.viridis,
                                 s=200, edgecolor='#ffffff', linewidth=4, zorder=10)
                    plt.title('Datapoints Projection #' +  str(colnum), size=4*side)
                
            if labels != [ ] and not self.color_ex:
                recs = []
                for i in class_assignment:
                    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_assignment[i]))

                plt.legend(recs, class_assignment.keys(), loc=(1.25,0), frameon=False, fontsize=2*side)

            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight', dpi=72)
            if show == True:
                plt.show()
            plt.clf()
        
        """ Print the x,y coordinates of bmus, useful for the clustering function. """
        
        return [[pos[0],pos[1]] for pos in bmu_list] 
        
        
    def cluster(self, array, clus_type='qthresh', cutoff=5, quant=0.2, percent=0.02, num_cl=8,\
                    save_file=True, file_type='dat', show=False, print_out=True, out_path='./'):
    
        """Clusters the data in a given array according to the SOM trained map.
            The clusters can also be plotted.

        Args:
            array (self.interface.num.array): An array containing datapoints to be clustered.
            clus_type (str, optional): The type of clustering to be applied, so far only quality threshold (qthresh) 
                algorithm is directly implemented, other algorithms require sklearn.
            cutoff (float, optional): Cutoff for the quality threshold algorithm. This also doubles as
                maximum distance of two points to be considered in the same cluster with DBSCAN.
            percent (float, optional): The percentile that defines the reference distance in density peak clustering (dpeak).
            num_cl (int, optional): The number of clusters for K-Means clustering
            quant (float, optional): Quantile used to calculate the bandwidth of the mean shift algorithm.
            save_file (bool, optional): Choose to save the resulting clusters in a text file.
            file_type (string, optional): Format of the file where the clusters will be saved (csv or dat)
            show (bool, optional): Choose to display the plot.
            print_out (bool, optional): Choose to save the plot to a file.
            out_path (str, optional): Path to the folder where data will be saved.
            
        Returns:
            (list of int): A nested list containing the clusters with indexes of the input array points.
            
        """

        """ Call project to first find the bmu for each array datapoint, but without producing any graph. """

        bmu_list = self.project(array, show=False, print_out=False)
        clusters = []

        if clus_type == 'qthresh':
            
            """ Cluster according to the quality threshold algorithm (slow!). """
    
            clusters = qt.quality_threshold(bmu_list, cutoff, self.PBC, self.net_height, self.net_width)

        elif clus_type == 'dpeak':

            """ Cluster according to the density peak algorithm. """

            clusters = dp.density_peak(bmu_list, PBC=self.PBC, net_height=self.net_height, net_width=self.net_width)

        elif clus_type in ['MeanShift', 'DBSCAN', 'KMeans']:
        
            """ Cluster according to algorithms implemented in sklearn. """
        
            if self.PBC == True:
                print("Warning: Only Quality Threshold and Density Peak clustering work with PBC")

            try:
                
                if clus_type == 'MeanShift':
                    bandwidth = self.interface.cluster_algo.estimate_bandwidth(self.interface.num.asarray(bmu_list), quantile=quant, n_samples=500)
                    cl =  self.interface.cluster_algo.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmu_list)
                
                if clus_type == 'DBSCAN':
                    cl = self.interface.cluster_algo.DBSCAN(eps=cutoff, min_samples=5).fit(bmu_list)     
                
                if clus_type == 'KMeans':
                    cl = self.interface.cluster_algo.KMeans(n_clusters=num_cl).fit(bmu_list)

                cl_labs = cl.labels_                 
                    
                for i in self.interface.num.unique(cl_labs):
                    cl_list = []
                    tmp_list = range(len(bmu_list))
                    for j,k in zip(tmp_list,cl_labs):
                        if i == k:
                            cl_list.append(j)
                    clusters.append(cl_list)     
            except:
                print(('Unexpected error: ', sys.exc_info()[0]))
                raise
        else:
            sys.exit("Error: unkown clustering algorithm " + clus_type)

        
        if save_file == True:
            with open(os.path.join(out_path,clus_type+'_clusters.'+file_type),
                      'w') as file:
                if file_type == 'csv':
                    separator = ','
                else: 
                    separator = ' '
                for line in clusters:
                    for id in line: file.write(str(id)+separator)
                    file.write('\n')
        
        if print_out==True or show==True:
            
            print_name = os.path.join(out_path,clus_type+'_clusters.png')
            
            fig, ax = plt.subplots()
            
            for i in range(len(clusters)):
                randCl = "#%06x" % random.randint(0, 0xFFFFFF)
                xc,yc  = [],[]
                for c in clusters[i]:
                    #again, invert y and x to be consistent with the previous maps
                    xc.append(bmu_list[int(c)][0])
                    yc.append(self.net_height-bmu_list[int(c)][1])    
                ax.scatter(xc, yc, color=randCl, label='cluster'+str(i))

            plt.gca().invert_yaxis()
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)           
            ax.set_title('Clusters')
            ax.axis('off')

            if print_out == True:
                plt.savefig(print_name, bbox_inches='tight', dpi=600)
            if show == True:
                plt.show()
            plt.clf()   
            
        return clusters

        
class SOMNode:

    """ Single Kohonen SOM Node class. """
    
    def __init__(self, x, y, num_weights, net_height, net_width, PBC, interface,
                wei_bounds=None, init_vec=None, wei_array=None,):
    
        """Initialise the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            num_weights (int): Length of the weights vector.
            net_height (int): Network height, needed for periodic boundary conditions (PBC)
            net_width (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            interface (Interface obj): CPU/GPU interface.
            wei_bounds(self.interface.num.array, optional): boundary values for the random initialization
                of the weights. Must be in the format [min_val, max_val]. 
                They are overwritten by 'init_vec'.
            init_vec (self.interface.num.array, optional): Array containing the two custom vectors (e.g. PCA)
                for the weights initalization.
            wei_array (self.interface.num.array, optional): Array containing the weights to give
                to the node if loaded from a file.

                
        """
    
        self.interface = interface
        
        self.PBC       = PBC
        self.pos       = self.interface.num.array(hx.coor_to_hex(x,y))
        self.weights   = []

        self.net_height = net_height
        self.net_width  = net_width

        if wei_array is not None:
            """ Load nodes's weights from file. """
            
            self.weights = wei_array

        elif init_vec is not None:
            """ Select uniformly in the space spanned by the custom vectors. """

            self.weights = (x-self.net_width/2)*2.0/self.net_width*init_vec[0] + \
                           (y-self.net_height/2)*2.0/self.net_height*init_vec[1]

        elif wei_bounds is not None:
            """ Select randomly in the space spanned by the data. """
            
            for i in range(num_weights):
                self.weights.append(random.random()*(wei_bounds[1][i]-wei_bounds[0][i])+wei_bounds[0][i])
       
        else: 
            """ Else return error. """

            sys.exit(('Error in the network weights initialization, make sure to provide random initalization boundaries,\
                        custom vectors, or load the weights from file.'))
   
        self.weights = self.interface.num.array(self.weights)

    def get_distance(self, vec):
    
        """Calculate the distance between the weights vector of the node and a given vector.
           DEPRECATED: this function will be removed in future versions, use SOMNet.get_bmu instead.

        Args:
            vec (self.interface.num.array): The vector from which the distance is calculated.
            
        Returns: 
            (float): The distance between the two weight vectors.
        """
    
        sum = 0
        if len(self.weights) == len(vec):
            for i in range(len(vec)):
                sum += (self.weights[i]-vec[i])*(self.weights[i]-vec[i])
            return sqrt(sum)
        else:
            sys.exit("Error: dimension of nodes != input data dimension!")

    def get_node_distance(self, node):
    
        """Calculate the distance within the network between the node and another node.

        Args:
            node (SOMNode): The node from which the distance is calculated.
            
        Returns:
            (float): The distance between the two nodes.
            
        """

        #to clean up

        if self.PBC == True:

            """ Hexagonal Periodic Boundary Conditions """

            offset = 0 if self.net_height % 2 == 0 else 0.5

            return  min([sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #right
                            sqrt((self.pos[0]-node.pos[0]+self.net_width)*(self.pos[0]-node.pos[0]+self.net_width)\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #bottom 
                            sqrt((self.pos[0]-node.pos[0]+offset)*(self.pos[0]-node.pos[0]+offset)\
                                +(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)),
                            #left
                            sqrt((self.pos[0]-node.pos[0]-self.net_width)*(self.pos[0]-node.pos[0]-self.net_width)\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #top 
                            sqrt((self.pos[0]-node.pos[0]-offset)*(self.pos[0]-node.pos[0]-offset)\
                                +(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4)),
                            #bottom right
                            sqrt((self.pos[0]-node.pos[0]+self.net_width+offset)*(self.pos[0]-node.pos[0]+self.net_width+offset)\
                                +(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)),
                            #bottom left
                            sqrt((self.pos[0]-node.pos[0]-self.net_width+offset)*(self.pos[0]-node.pos[0]-self.net_width+offset)\
                                +(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.net_height*2/sqrt(3)*3/4)),
                            #top right
                            sqrt((self.pos[0]-node.pos[0]+self.net_width-offset)*(self.pos[0]-node.pos[0]+self.net_width-offset)\
                                +(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4)),
                            #top left
                            sqrt((self.pos[0]-node.pos[0]-self.net_width-offset)*(self.pos[0]-node.pos[0]-self.net_width-offset)\
                                +(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.net_height*2/sqrt(3)*3/4))])
                        
        else:
            return sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])\
                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1]))



    def update_weights(self, input_vec, sigma, learning_rate, bmu):
    
        """Update the node weights.

        Args:
            input_vec (self.interface.num.array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            learning_rate (float): The updated learning rate.
            bmu (SOMNode): The best matching unit.
        """
    
        dist  = self.get_node_distance(bmu)
        gauss = exp(-dist*dist/(2*sigma*sigma))

        i=0
        
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - gauss*learning_rate*(self.weights[i]-input_vec[i])
        
        
if __name__ == "__main__":

    pass
