import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

coo = np.ones((10,3))
coo2 = 2*np.ones((10,3))
print(coo.shape)
print(coo[:, 1])
print(coo[:,1].shape)
print(coo[:, 1]*coo2[:,2])
print((coo2[:,2]**2).shape)
print(coo2[:,0:3])