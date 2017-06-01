"""
#########################################################################
Density-Peak Clustering

A Rodriguez, A Laio,
Clustering by fast search and find of density peaks
SCIENCE, 1492, vol 322 (2014) 

F. Comitani @2017 
#########################################################################

WARNING: Work in Progress... Not tested!

"""

import sys
import numpy as np
from operator import attrgetter
import warnings
import matplotlib.pyplot as plt

class pt:
	""" Class for the points to cluster. """

	def __init__(self, coordinates):

		"""Initialise the point.

		Args:
			coordinates (np.array): Array containing the point coordinates in N dimensions.

		"""
		
		self.coor=[]
		for c in coordinates:
			self.coor.append(c)

		""" Initialise empty density (rho), higher-density distance (delta) and list of distances."""
		self.rho=0
		self.delta=sys.maxsize
		self.dists=[]

	def set_dist(self, coll):
	
	 	"""Calculate the distances from all other points in a collection. 

	 	Args:
			coll (collection): collection containing all the points of the dataset used to calculate the distances. """

	 	for p2 in coll.points:
	 			if self!=p2: self.dists.append(dist(self,p2))


	def set_rho(self, coll, typeFunc='step'):

		"""Calculate the density of the single point for a given dataset. [Deprecated]

		Args:
			coll (collection): collection containing all the points of the dataset used to calculate the density. 
			typeFunc (str): step function type (step, gaussian kernel or logistic).

		"""

		warnings.warn('Setting individual rhos is deprecated, use the collection.set_rhos() instead!', DeprecationWarning)

		if self not in coll.points:
			print('WARNING: calculating the density for a point that was not found in the dataset, make sure to be consistent with your data!')
		
		for p2 in coll.points:
			if self!=p2: 
				if typeFunc=='step':
					self.rho=self.rho+step(self,p2,self.refd)
				elif typeFunc=='gaussian':
					self.rho=self.rho+step(self,p2,self.refd)
				elif typeFunc=='logistic':
					self.rho=self.rho+step(self,p2,self.refd)
				else:
					""" Raise exception if metric other then euclidean is used. """
					raise NotImplementedError('Only step, gaussian kernel or logistic functions are implemented')

	def set_delta(self, coll):

		"""Calculate the distance of the point from higher density points. [Deprecated]

		Args:
			coll (collection): collection containing all the points of the dataset used to calculate the distance.

		"""
		
		warnings.warn('Setting individual deltas is deprecated, use the collection.set_deltas() instead!', DeprecationWarning)
		
		if self not in coll.points:
			print('WARNING: calculating the distance for a point that was not found in the dataset, make sure to be consistent with your data!')

		mind=sys.maxsize
		distHigh, distLow= [], []
		
		for p2 in coll.points:
			if self!=p2:
				if self.rho<p2.rho:
					distHigh.append(dist(self,p2))
				else:
					distsLow.append(dist(self,p2))
		
		""" If the point has maximal rho, then return max distance """

		if len(distHigh)>0: self.delta=np.min(distHigh)
		else: self.delta=np.max(distLow)


class collection:

	"""Class for a collection of point objects. """

	def __init__(self, coorArray, typeFunc='gaussian', percent=0.2):
		
		"""	Generate a collection of point objects from an array containing their coordinates.
	
		Args:
			coorArray (np.array): Array containing the coordinates of the points to cluster.
			typeFunc (str): step function for calculating rho (step, gaussian kernel or logistic)
			percent (float): average percentage of neighbours

		"""
	
		self.points=[]
		for coors in coorArray:
			self.points.append(pt(coors))	

		#Not sure about this... quintile of distances?
		##########CHECK#############################

		self.set_dists()
		#############fix this#####################
		index=int(np.round(len(self.points)*percent))

		alldists=[]
		for p1 in self.points:
			for p2 in self.points:
				if p1<p2: alldists.append(dist(p1,p2))

		alldists.sort()
		self.refd=alldists[index]

		""" Make sure rhos are set before setting deltas """

		self.set_rhos(typeFunc)
		self.set_deltas()	


	def set_dists(self):
	
	 	"""Calculate the distance matrix for all points. """

	 	for p1 in self.points:
	 		for p2 in self.points:
	 			if p1<p2: 
	 				d=dist(p1,p2)
	 				p1.dists.append(d), p2.dists.append(d)


	def set_rhos(self, typeFunc='step'):
	
		"""Calculate the density for each point in the dataset. 

		Args:
			typeFunc (str): step function type (step, gaussian or logistic)

		"""

		for p1 in self.points:
			for p2 in self.points:
				if p1<p2: 
					if typeFunc=='step':
						p1.rho=p1.rho+step(p1,p2,self.refd)
						p2.rho=p2.rho+step(p1,p2,self.refd)
					elif typeFunc=='gaussian':
						p1.rho=p1.rho+gaussian(p1,p2,self.refd)
						p2.rho=p2.rho+gaussian(p1,p2,self.refd)
					elif typeFunc=='logistic':
						p1.rho=p1.rho+sigmoid(p1,p2,self.refd)
						p2.rho=p2.rho+sigmoid(p1,p2,self.refd)
					else:
						""" Raise exception if metric other then euclidean is used """
						raise NotImplementedError('Only step, gaussian kernel or logistic functions are implemented')

	def set_deltas(self):

		"""Calculate the distance from higher density points for each point in the dataset. """

		distHigh, distLow= [], []
		for p1 in self.points:
			for p2 in self.points:
			 	# No need to re-set the distances
				d=dist(p1,p2)
				#p1.dists.append(d), p2.dists.append(d)
				if p1<p2: 
					if p1.rho<p2.rho and d<p1.delta: p1.delta=d
					elif p1.rho>p2.rho and d<p2.delta: p2.delta=d
	
		""" If the point has maximal rho, then return max distance """

		pmax=max(self.points, key=attrgetter('rho'))
		pmax.set_dist(self)
		pmax.rho=max(pmax.dists)


def dist(p1,p2, metric='euclid'):

	"""Calculate the distance between two point objects in a N dimensional space according to a given metric.

	Args:
		p1 (point): First point object for the distance.
		p2 (point): Second point object for the distance.
		metric (string): Metric to use. For now only euclidean distance is implemented.

	Returns:
		float): The distance between the two points.

	"""

	if metric=='euclid':
		if len(p1.coor)!=len(p2.coor): raise ValueError('Points must have the same dimensionality!')
		else:
			diffs=0
			for i in range(len(p1.coor)): 
				diffs=diffs+((p1.coor[i]-p2.coor[i])*(p1.coor[i]-p2.coor[i]))
				return np.sqrt(diffs)
	else:
		""" Raise exception if metric other then euclidean is used """
		raise NotImplementedError('Only euclidean metric is implemented')


def step(p1, p2, cutoff):

	"""Step function activated when the distance of two points is less than the cutoff.

	Args:
		p1 (point): First point object for the distance.
		p2 (point): Second point object for the distance.
		cutoff (float): The cutoff to define the proximity of the points.

	Returns:
		(int): 1 if the points are closer than the cutoff, 0 otherwise.

	"""

	if dist(p1,p2)<cutoff: return 1
	else: return 0	

def gaussian(p1, p2, sigma):

	"""Gaussian function of the distance between two points scaled with sigma.

	Args:
		p1 (point): First point object for the distance.
		p2 (point): Second point object for the distance.
		sigma (float): The scaling factor for the distance.

	Returns:
		(float): value of the gaussian function.

	"""

	return np.exp(-1.0*dist(p1,p2)*dist(p1,p2)/sigma*sigma)

def sigmoid(p1, p2, sigma):

	"""Logistic function of the distance between two points scaled with sigma.

	Args:
		p1 (point): First point object for the distance.
		p2 (point): Second point object for the distance.
		sigma (float): The scaling factor for the distance.

	Returns:
		(float): value of the logistic function.

	"""

	return np.exp(-1.0*(1.0+np.exp((dist(p1,p2))/sigma)))



if __name__ == "__main__":

  	print("Testing...")
	samples1 = np.random.multivariate_normal([0, 0], [[1, 0.1],[0.1, 1]], 100)
	samples2 = np.random.multivariate_normal([10, 10], [[2, 0.5],[0.5, 2]], 100)
	samples = np.concatenate((samples1,samples2), axis=0)
#	plt.plot(samples[:, 0], samples[:, 1], '.')
#	plt.show()

  	pts=collection(samples)
  	print samples[0]
  	print pts.points[0].coor
  	print pts.points[0].delta
  	print pts.points[0].rho


