"""
Quality Threshold Clustering

L. J. Heyer, S. Kruglyak and S. Yooseph, 
Exploring Expression Data: Identification and Analysis of Coex-pressed Genes 
Genome Research, Vol. 9, No. 11, 1999, pp. 1106-1115. 

F. Comitani @2017 
"""

import numpy as np

def qualityThreshold(sample, cutoff=5, PBC=False, netHeight=0, netWidth=0):

	""" Run the complete clustering algorithm in one go and returns the clustered indeces as a list.

		Args:
			sample (array): The input dataset
			cutoff (float, optional): The clustering cutoff distance.
			PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
			netHeight (int, optional): Number of nodes along the first dimension, required for PBC.
			netWidth (int, optional): Numer of nodes along the second dimension, required for PBC.
			

		Returns:
			clusters (list, int): a list of lists containing the points indices belonging to each cluster
	"""		

	tmpList=list(range(len(sample)))
	clusters=[]

	while len(tmpList)!=0:
		
		qtList=[]	
		for i in tmpList:
			qtList.append([])
			for j in tmpList:
		
				if PBC is True:
					""" Hexagonal Periodic Boundary Conditions """
				
					if netHeight%2==0:
						offset=0
					else: 
						offset=0.5
				 
					distBmu=np.min([np.sqrt((sample[j][0]-sample[i][0])*(sample[j][0]-sample[i][0])\
						+(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
					#right
					np.sqrt((sample[j][0]-sample[i][0]+netWidth)*(sample[j][0]-sample[i][0]+netWidth)\
						+(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
					#bottom 
					np.sqrt((sample[j][0]-sample[i][0]+offset)*(sample[j][0]-sample[i][0]+offset)\
						+(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)),
					#left
					np.sqrt((sample[j][0]-sample[i][0]-netWidth)*(sample[j][0]-sample[i][0]-netWidth)\
						+(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
					#top 
					np.sqrt((sample[j][0]-sample[i][0]-offset)*(sample[j][0]-sample[i][0]-offset)\
						+(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4)),
					#bottom right
					np.sqrt((sample[j][0]-sample[i][0]+netWidth+offset)*(sample[j][0]-sample[i][0]+netWidth+offset)\
						+(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)),
					#bottom left
					np.sqrt((sample[j][0]-sample[i][0]-netWidth+offset)*(sample[j][0]-sample[i][0]-netWidth+offset)\
						+(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+netHeight*2/np.sqrt(3)*3/4)),
					#top right
					np.sqrt((sample[j][0]-sample[i][0]+netWidth-offset)*(sample[j][0]-sample[i][0]+netWidth-offset)\
						+(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4)),
					#top left
					np.sqrt((sample[j][0]-sample[i][0]-netWidth-offset)*(sample[j][0]-sample[i][0]-netWidth-offset)\
						+(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-netHeight*2/np.sqrt(3)*3/4))])
				
				else:
					distBmu=np.sqrt((sample[j][0]-sample[i][0])*(sample[j][0]-sample[i][0])\
					+(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1]))

				if distBmu <= cutoff:
					qtList[-1].append(j)

		clusters.append(max(qtList,key=len))
		for el in clusters[-1]:
			tmpList.remove(el)

	return clusters


def test():

	import matplotlib.pyplot as plt
	import matplotlib as mpl


	""" Run the complete clustering algorithm on a test case and print the clustered points graph. """

	print("Testing...")

	np.random.seed(100)
	samples1 = np.random.multivariate_normal([0, 0], [[1, 0.1],[0.1, 1]], 100)
	samples2 = np.random.multivariate_normal([10, 10], [[2, 0.5],[0.5, 2]], 100)
	samples3 = np.random.multivariate_normal([0, 10], [[2, 0.5],[0.5, 2]], 100)
	samples4 = np.random.uniform(0, 14, [50,2])
	samplesTmp = np.concatenate((samples1,samples2), axis=0)
	samplesTmp2 = np.concatenate((samplesTmp,samples3), axis=0)
	samples = np.concatenate((samplesTmp2,samples4), axis=0)
#	plt.plot(samples[:, 0], samples[:, 1], '.')
#	plt.show()

	clusters = qualityThreshold(samples, cutoff=5)

	for c in clusters:
		plt.plot([samples[i][0] for i in c], [samples[i][1] for i in c], 'o', c="#%06x" % np.random.randint(0, 0xFFFFFF))

	plt.show()
	
	print("Done!")


if __name__ == "__main__":

	test()
