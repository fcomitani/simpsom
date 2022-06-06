import sys, os, time
sys.path.append("../../simpsom")
import simpsom as sps
import pandas as pd
import numpy as np
from datetime import datetime

i, j       = 1000, 100
x          = 100
epochs     = 100
data       = np.random.rand(i, j)



start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="hexagonal")
net.train(train_algo='online', epochs=1000)
end   = time.time()
elaps = end-start
print("\t *** topology='hexagonal', GPU=True, train_algo='online', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="hexagonal")
net.train(train_algo='batch', epochs=epochs, batch_size=-1) 
end   = time.time()
elaps = end-start
print("\t *** topology='hexagonal', GPU=True, train_algo='batch', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="hexagonal")
net.train(train_algo='batch', epochs=epochs, batch_size=-1, early_stop='mapdiff') 
end   = time.time()
elaps = end-start
print("\t *** topology='hexagonal', GPU=True, train_algo='batch', early_stop='mapdiff', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="hexagonal")
net.train(train_algo='batch', epochs=epochs, batch_size=-1, early_stop='bmudiff') 
end   = time.time()
elaps = end-start
print("\t *** topology='hexagonal', GPU=True, train_algo='batch', early_stop='bmudiff', elapsed_time=%.5f"%elaps)



start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="rectangular")
net.train(train_algo='online', epochs=1000)
end   = time.time()
elaps = end-start
print("\t *** topology='rectangular', GPU=True, train_algo='online', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="rectangular")
net.train(train_algo='batch', epochs=epochs, batch_size=-1) 
end   = time.time()
elaps = end-start
print("\t *** topology='rectangular', GPU=True, train_algo='batch', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="rectangular")
net.train(train_algo='batch', epochs=epochs, batch_size=-1, early_stop='mapdiff') 
end   = time.time()
elaps = end-start
print("\t *** topology='rectangular', GPU=True, train_algo='batch', early_stop='mapdiff', elapsed_time=%.5f"%elaps)

start = time.time()
net   = sps.SOMNet(x, x, data, GPU=True, topology="rectangular")
net.train(train_algo='batch', epochs=epochs, batch_size=-1, early_stop='bmudiff') 
end   = time.time()
elaps = end-start
print("\t *** topology='rectangular', GPU=True, train_algo='batch', early_stop='bmudiff', elapsed_time=%.5f"%elaps)


