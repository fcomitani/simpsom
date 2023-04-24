from xpysom import XPySom    

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

som = XPySom(3, 3, 4, sigma=0.3, learning_rate=0.5, random_seed=42) # initialization of 6x6 SOM
som.train_batch(data, 5) # trains the SOM with 5 iterations

print(som.get_weights())