import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

path = "C:\E\\USTC\MagLab\\20191105purenoise"
os.chdir(path)
fnum = 10

fname = 1825
length = 90000
arr = np.zeros((length * fnum), dtype=float)
for i in range( fnum ):
    arr[i * length: (i+1) * length] = np.loadtxt("%4d"%(fname + i) + ".txt").flatten()[1::2]
#print(arr[1:10].flatten()[0::2])
plt.hist(arr,bins=200)
plt.show()