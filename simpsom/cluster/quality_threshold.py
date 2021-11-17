"""
Quality Threshold Clustering

L. J. Heyer, S. Kruglyak and S. Yooseph, 
Exploring Expression Data: Identification and Analysis of Coex-pressed Genes 
Genome Research, Vol. 9, No. 11, 1999, pp. 1106-1115. 

F. Comitani @2017 
"""

import numpy as np

def quality_threshold(sample, cutoff=5, PBC=False, net_height=0, net_width=0):

    """ Run the complete clustering algorithm in one go and returns the clustered indices as a list.

        Args:
            sample (array): The input dataset
            cutoff (float, optional): The clustering cutoff distance.
            PBC (bool, optional): Activate/deactivate Periodic Boundary Conditions.
            net_height (int, optional): Number of nodes along the first dimension, required for PBC.
            net_width (int, optional): Numer of nodes along the second dimension, required for PBC.
            

        Returns:
            clusters (list, int): a list of lists containing the points indices belonging to each cluster
    """		

    tmp_list = list(range(len(sample)))
    clusters = []

    while len(tmp_list) != 0:
        
        qt_list = []	
        for i in tmp_list:
            qt_list.append([])
            for j in tmp_list:
        
                if PBC is True:
                    """ Hexagonal Periodic Boundary Conditions """
                
                    offset = 0 if net_height % 2 == 0 else 0.5

                 
                    dist_bmu = np.min([np.sqrt((sample[j][0]-sample[i][0])*(sample[j][0]-sample[i][0])\
                        +(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
                    #right
                    np.sqrt((sample[j][0]-sample[i][0]+net_width)*(sample[j][0]-sample[i][0]+net_width)\
                        +(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
                    #bottom 
                    np.sqrt((sample[j][0]-sample[i][0]+offset)*(sample[j][0]-sample[i][0]+offset)\
                        +(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)),
                    #left
                    np.sqrt((sample[j][0]-sample[i][0]-net_width)*(sample[j][0]-sample[i][0]-net_width)\
                        +(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1])),
                    #top 
                    np.sqrt((sample[j][0]-sample[i][0]-offset)*(sample[j][0]-sample[i][0]-offset)\
                        +(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4)),
                    #bottom right
                    np.sqrt((sample[j][0]-sample[i][0]+net_width+offset)*(sample[j][0]-sample[i][0]+net_width+offset)\
                        +(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)),
                    #bottom left
                    np.sqrt((sample[j][0]-sample[i][0]-net_width+offset)*(sample[j][0]-sample[i][0]-net_width+offset)\
                        +(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]+net_height*2/np.sqrt(3)*3/4)),
                    #top right
                    np.sqrt((sample[j][0]-sample[i][0]+net_width-offset)*(sample[j][0]-sample[i][0]+net_width-offset)\
                        +(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4)),
                    #top left
                    np.sqrt((sample[j][0]-sample[i][0]-net_width-offset)*(sample[j][0]-sample[i][0]-net_width-offset)\
                        +(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4)*(sample[j][1]-sample[i][1]-net_height*2/np.sqrt(3)*3/4))])
                
                else:
                    dist_bmu = np.sqrt((sample[j][0]-sample[i][0])*(sample[j][0]-sample[i][0])\
                    +(sample[j][1]-sample[i][1])*(sample[j][1]-sample[i][1]))

                if dist_bmu <= cutoff:
                    qt_list[-1].append(j)

        clusters.append(max(qt_list,key=len))
        for el in clusters[-1]:
            tmp_list.remove(el)

    return clusters


def qt_test(out_path='./'):

    import os
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    """ Run the Quality Threshold on a test caseand print the clustered points graph. 
    
        Args:
            out_path (str, optional): path to the output folder.
    """
    
    """ Set up output folder. """

    if out_path != './':
        try:
            os.makedirs(out_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    print("Testing Quality Threshold...")

    np.random.seed(100)
    samples1     = np.random.multivariate_normal([0, 0], [[1, 0.1],[0.1, 1]], 100)
    samples2     = np.random.multivariate_normal([10, 10], [[2, 0.5],[0.5, 2]], 100)
    samples3     = np.random.multivariate_normal([0, 10], [[2, 0.5],[0.5, 2]], 100)
    samples4     = np.random.uniform(0, 14, [50,2])
    samples_tmp  = np.concatenate((samples1,samples2), axis=0)
    samples_tmp2 = np.concatenate((samples_tmp,samples3), axis=0)
    samples      = np.concatenate((samples_tmp2,samples4), axis=0)

    clusters = quality_threshold(samples, cutoff=5)

    for c in clusters:
        plt.plot([samples[i][0] for i in c], [samples[i][1] for i in c], 'o', c="#%06x" % np.random.randint(0, 0xFFFFFF))

    plt.savefig(os.path.join(out_path,'qt_test_out.png'), bbox_inches='tight', dpi=200)
    
    print("Done!")

if __name__ == "__main__":

    test()
