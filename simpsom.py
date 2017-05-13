#########################################################################
################ SimpSOM (Simple Self Organizing Maps) ##################
############################### v1.1.0 ##################################
######################### F. Comitani @2017 #############################
#########################################################################


import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

class somNet:
	""" Kohoen SOM Network class. """

	def __init__(self, netHeight, netWidth, data, loadFile=None):

		"""Initialise the SOM network.

		Args:
			netHeight (int): Number of nodes along the first dimension.
			netWidth (int): Numer of nodes along the second dimension.
			data (np.array or list): N-dimensional dataset.
			loadFile (str, optional): Name of file to load containing information 
				to initialise the network weights.
				
		"""
	
		""" Switch to activate special workflows if running the colours example. """
		self.colorEx=False

		self.nodeList=[]
		self.data=data.reshape(np.array([data.shape[0], data.shape[1]]))

		""" Load the weights from file or generate them randomly. """
		
		if loadFile==None:
			self.netHeight = netHeight
			self.netWidth = netWidth

			for x in range(self.netWidth):
				for y in range(self.netHeight):
					self.nodeList.append(somNode(x,y, self.data.shape[1]))
		else: 	
			if loadFile.endswith('.npy')==False:
				loadFile=loadFile+'.npy'
			weiArray=np.load(loadFile)
			#add something to check that data and array have the same dimensions,
			#or that they are mutually exclusive
			self.netHeight = int(weiArray[0][0])
			self.netWidth = int(weiArray[0][1])

			#start from 1 because 0 contains info on the size of the network
			countWei=1
			for x in range(self.netWidth):
				for y in range(self.netHeight):
					self.nodeList.append(somNode(x,y, self.data.shape[1], weiArray[countWei]))
					countWei+=1

	def save(self, fileName='somNet_trained'):
	
		"""Saves the nodes weights to a file.

		Args:
			fileName (str, optional): Name of file where the weights will be saved.
			
		"""
		
		#save network dimensions
		weiArray=[np.zeros(len(self.nodeList[0].weights))]
		weiArray[0][0],weiArray[0][1]=self.netHeight, self.netWidth
		#save the weights
		for node in self.nodeList:
			weiArray.append(node.weights)
		np.save(fileName, np.asarray(weiArray))
	

	def update_sigma(self, iter):
	
		"""Update the gaussian sigma.

		Args:			
			iter (int): Iteration number.
			
		"""
	
		self.sigma = self.startSigma * np.exp(-iter/self.tau);
	
	def update_lrate(self, iter):
	
		"""Update the learning rate.

		Args:			
			iter (int): Iteration number.
			
		"""
		
		self.lrate =  self.startLearnRate * np.exp(-iter/self.epochs);
	
	def find_bmu(self, vec):
	
		"""Find the best matching unit (BMU) for a given vector.

		Args:			
			vec (np.array): The vector to match.
			
		Returns:			
			bmu (somNode): The best matching unit node.
			
		"""
	
		minVal=np.iinfo(np.int).max
		for node in self.nodeList:
			dist=node.get_distance(vec)
			if dist < minVal:
				minVal=dist
				bmu=node
		return bmu	
			
	def train(self, epochs=5000, startLearnRate=0.01):
	
		"""Train the SOM.

		Args:
			epochs (int): Number of training iterations.
			startLearnRate (float): Initial learning rate.
			
		"""
		
		print "Training SOM... 0%",

		self.startSigma = max(self.netHeight, self.netWidth)/2
		self.startLearnRate = startLearnRate
		self.epochs=epochs
		self.tau = self.epochs/np.log(self.startSigma)
	
		for i in range(self.epochs):

			if i%100==0:
				print "\rTraining SOM... "+str(int(i*100.0/self.epochs))+"%" ,

			self.update_sigma(i)
			self.update_lrate(i)
			
			""" Train with the random point method: 
				instead of using all the training points, a random datapoint is chosen
				for each iteration and used to update the weights of all the nodes.
			"""
			
			inputVec = self.data[np.random.randint(0, self.data.shape[0]), :].reshape(np.array([self.data.shape[1], 1]))
			
			bmu=self.find_bmu(inputVec)
			
			for node in self.nodeList:
				node.update_weights(inputVec, self.sigma, self.lrate, bmu)

		print "\rTraining SOM... done!"

		
	def nodes_graph(self, colnum=0, show=False, printout=True):
	
		"""Plot a 2D map with nodes and weights values

		Args:
			colnum (int): The index of the weight that will be shown as colormap.
			show (bool, optional): Choose to display the plot.
			printout (bool, optional): Choose to save the plot to a file.
			
		"""
	
		fig, ax = plt.subplots()
		
		if self.colorEx==True:
			cols = [(node.weights[0],node.weights[1],node.weights[2]) for node in self.nodeList]
			cols=np.asarray(cols).reshape((self.netHeight, self.netWidth, len(cols[0])))
			cax=ax.imshow(cols, interpolation='none')
			ax.set_title('Node Grid w Color Features')
			printName='nodesColors.png'
		else:
			cols = [node.weights[colnum] for node in self.nodeList]
			cols=np.asarray(cols).reshape((self.netHeight, self.netWidth))
			cax=ax.imshow(cols, interpolation='none', cmap=cm.viridis)
			cbar=fig.colorbar(cax)
			cbar.set_label('Feature #' +  str(colnum)+' value')
			ax.set_title('Node Grid w Feature #' +  str(colnum))
			printName='nodesFeature_'+str(colnum)+'.png'

		if printout==True:
			plt.savefig(printName, bbox_inches='tight', dpi=600)
		if show==True:
			plt.show()
		if show!=False and printout!=False:
			plt.clf()
					 
	def diff_graph(self, show=False, printout=True):
	
		"""Plot a 2D map with nodes and weights difference among neighbouring nodes

		Args:
			show (bool, optional): Choose to display the plot.
			printout (bool, optional): Choose to save the plot to a file.
			
		"""
		
		neighbours=[]
		for node in self.nodeList:
			nodelist=[]
			for nodet in self.nodeList:
				if node != nodet and node.get_nodeDistance(nodet) <= np.sqrt(2):
					nodelist.append(nodet)
			neighbours.append(nodelist)		
			
		diffs = np.zeros((self.netHeight, self.netWidth))
		for node, neighbours in zip(self.nodeList, neighbours):
			diff=0
			for nb in neighbours:
				diff=diff+node.get_distance(nb.weights)
			diffs[node.pos[0],node.pos[1]]=diff	

		fig, ax = plt.subplots()
		cax=ax.imshow(diffs, interpolation='none', cmap=cm.Blues)
		cbar=fig.colorbar(cax)

		cbar.set_label('Weights Difference')
		ax.set_title('Nodes Grid w Weights Difference ')

		if printout==True:
			plt.savefig('nodesDifference.png', bbox_inches='tight', dpi=600)
		if show==True:
			plt.show()
		plt.clf()

	def project(self, array, colnum=0, labels=[], show=False, printout=True):

		"""Project the datapoints of a given array to the 2D space of the 
			SOM by calculating the bmus. If requested plot a 2D map with as 
			implemented in nodes_graph and adds circles to the bmu
			of each datapoint in a given array.

		Args:
			array (np.array): An array containing datapoints to be mapped.
			colnum (int): The index of the weight that will be shown as colormap.
			show (bool, optional): Choose to display the plot.
			printout (bool, optional): Choose to save the plot to a file.
			
		Returns:
			(list): bmu x,y position for each input array datapoint. 
			
		"""

		bmuList,cls=[],[]
		for i in range(array.shape[0]):
			bmuList.append(self.find_bmu(array[i,:]).pos)	
			if self.colorEx==True:
				cls.append(array[i,:])
			else: 
				cls.append(array[i,colnum])

		if show==True or printout==True:
		
			""" Call nodes_graph to first build the 2D map of the nodes. """
		
			self.nodes_graph(colnum, False, False)

			#be careful, x,y are inverted in nodes_graph imshow()		
			if self.colorEx==True:
				printName='colorProjection.png'
				plt.scatter([pos[1] for pos in bmuList],[pos[0] for pos in bmuList], color=cls, edgecolor='black')
				plt.title('Datapoints Projection')
			else:
				printName='projection_'+str(colnum)+'.png'
				plt.scatter([pos[1] for pos in bmuList],[pos[0] for pos in bmuList], color=cls, edgecolor='black', cmap=cm.viridis)
				plt.title('Datapoints Projection #' +  str(colnum))

			if labels!=[]:
				for label, x, y in zip(labels, [pos[1] for pos in bmuList],[pos[0] for pos in bmuList]):
					plt.annotate(label, xy=(x,y), xytext=(-0.25, 0.25), textcoords='offset points', ha='right', va='bottom') 
			

			if printout==True:
				plt.savefig(printName, bbox_inches='tight', dpi=600)
			if show==True:
				plt.show()
			plt.clf()
		
		""" Print the x,y coordinates of bmus, useful for the clustering function. """
		
		return [[pos[0],pos[1]] for pos in bmuList] 
		
		
	def cluster(self, array, type='qtresh', cutoff=5, quant=0.2, \
					savefile=True, show=False, printout=True):
	
		"""Clusters the data in a given array according to the SOM trained map.
			The clusters can also be plotted.

		Args:
			array (np.array): An array containing datapoints to be clustered.
			type (str, optional): The type of clustering to be applied, so far only quality treshold (qtresh) 
				algorithm is directly implemented, other algorithms require sklearn.
			cutoff(float, optional): Cutoff for the quality treshold algorithm. This also doubles as
				maximum distance of two points to be considered in the same cluster with DBSCAN.
			quant(float, optional): Quantile used to calculate the bandwidth of the mean shift algorithm.
			mdist(float, optional): 
			saveFile (bool, optional): Choose to save the resulting clusters in a text file.
			show (bool, optional): Choose to display the plot.
			printout (bool, optional): Choose to save the plot to a file.
			
		Returns:
			(list of int): A nested list containing the clusters with indexes of the input array points.
			
		"""

		""" Call project to first find the bmu for each array datapoint, but without producing any graph. """

		bmuList = self.project(array, show=False, printout=False)
		clusters=[]

		if type=='qtresh':
			
			""" Cluster according to the quality treshold algorithm (slow!). """
	
			tmpList=range(len(bmuList))
			while len(tmpList)!=0:
				qtList=[]
				#the cluster list for each point contains the point itself
				#so that it gets automatically
				#deleted at the next loop without having to worry about it
				for i in tmpList:
					qtList.append([])
					for j in tmpList:
						if np.sqrt((bmuList[j][0]-bmuList[i][0])*(bmuList[j][0]-bmuList[i][0])\
							+(bmuList[j][1]-bmuList[i][1])*(bmuList[j][1]-bmuList[i][1])) <= cutoff:
							qtList[-1].append(j)
				clusters.append(max(qtList,key=len))
				for el in clusters[-1]:
					tmpList.remove(el)

		else:
		
			""" Cluster according to algorithms implemented in sklearn. """
			#TODO: for now only MeanShift and DBSCAN are implemented, add the rest of the algorithms	
		
			try:
				#TODO: clean all this mess
				from sklearn import cluster
		
				if type=='MeanShift':
					bandwidth = cluster.estimate_bandwidth(np.asarray(bmuList), quantile=quant, n_samples=500)
					cl = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmuList)
				
				if type=='DBSCAN':
					cl = cluster.DBSCAN(eps=cutoff, min_samples=5).fit(bmuList)		
				
				clLabs = cl.labels_					
					
				for i in np.unique(clLabs):
					clList=[]
					tmpList=range(len(bmuList))
					for j,k in zip(tmpList,clLabs):
						if i==k:
							clList.append(j)
					clusters.append(clList)		
			except:
				print 'Unexpected error: ', sys.exc_info()[0]
				raise
		
		if savefile==True:
			file=open(type+'_clusters.txt', 'w')
			for line in clusters:
				for id in line: file.write(str(id)+' ')
				file.write('\n')
			file.close()
		
		if printout==True or show==True:
			
			np.random.seed(0)
			printName=type+'_clusters.png'
			
			fig, ax = plt.subplots()
			
			for i in range(len(clusters)):
				randCl = "#%06x" % np.random.randint(0, 0xFFFFFF)
				xc,yc=[],[]
				for c in clusters[i]:
					#again, invert y and x to be consistent with the previous maps
					xc.append(bmuList[int(c)][1])
					yc.append(bmuList[int(c)][0])	
				ax.scatter(xc, yc, color=randCl, label='cluster'+str(i))

			plt.gca().invert_yaxis()
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)			
			ax.set_title('Clusters')

			if printout==True:
				plt.savefig(printName, bbox_inches='tight', dpi=600)
			if show==True:
				plt.show()
			plt.clf()	
			
		return clusters

		
class somNode:

	""" Single Kohoen SOM Node class. """
	
	def __init__(self, x, y, numWeights, weiArray=[]):
	
		"""Initialise the SOM node.

		Args:
			x (int): Position along the first network dimension.
			y (int): Position along the second network dimension
			numWeights (int): Length of the weights vector.
			weiArray (np.array, optional): Array containing the weights to give
				to the node if a file was loaded.
				
		"""
	
		self.pos = [x,y]
		self.weights = []
		
		if weiArray==[]:
			for i in range(numWeights):
				self.weights.append(np.random.random())
		else:
			for i in range(numWeights):
				self.weights.append(weiArray[i])

	
	def get_distance(self, vec):
	
		"""Calculate the distance between the weights vector of the node and a given vector.

		Args:
			vec (np.array): The vector from which the distance is calculated.
			
		Returns: 
			(float): The distance between the two weight vectors.
		"""
	
		sum=0
		if len(self.weights)==len(vec):
			for i in range(len(vec)):
				sum+=(self.weights[i]-vec[i])*(self.weights[i]-vec[i])
			return np.sqrt(sum)
		else:
			sys.exit("Error: dimension of nodes != input data dimension!")

	def get_nodeDistance(self, node):
	
		"""Calculate the distance within the network between the node and another node.

		Args:
			node (somNode): The node from which the distance is calculated.
			
		Returns:
			(float): The distance between the two nodes.
			
		"""
	
		return np.sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])\
			+(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1]))		
		
	def update_weights(self, inputVec, sigma, lrate, bmu):
	
		"""Update the node Weights.

		Args:
			inputVec (np.array): A weights vector whose distance drives the direction of the update.
			sigma (float): The updated gaussian sigma.
			lrate (float): The updated learning rate.
			bmu (somNode): The best matching unit.
		"""
	
		dist=self.get_nodeDistance(bmu)
		gauss=np.exp(-dist*dist/(2*sigma*sigma))
		if gauss>0:
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] - gauss*lrate*(self.weights[i]-inputVec[i])

		
if __name__ == "__main__":	

	"""Here an example of the usage of the library is run: a number of vectors of length three
		(corresponding to the RGB values of a color) are used to briefly train a small network.
		Different example graphs are then printed from the trained network.		
	"""	

	#np.random.seed(0)
	#raw_data = np.random.random((20, 3))
	raw_data =np.asarray([[1, 0, 0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.2,0.2,0.5]])
	labels=['red','green','blue','yellow','magenta','cyan','indigo']

	print "Welcome to SimpSOM (Simple Self Organizing Maps) v1.0.0!\nHere is a quick example of what this library can do.\n"
	print "The algorithm will now try to map the following colors: ",
	for i in range(len(labels)-1):
			print labels[i] + ", ", 
	print "and " + labels[-1]+ ".\n"
	
	net = somNet(20, 20, raw_data)
	net.colorEx=True
	net.train(10000, 0.01)

	print "Saving weights and a few graphs...",
	net.save('colorExample_weights')
	net.nodes_graph()
	net.diff_graph()
	net.project(raw_data, labels=labels)
	
	print "done!"