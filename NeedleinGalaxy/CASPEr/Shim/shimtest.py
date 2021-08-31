import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

remax = np.amax(arr)
histct = np.zeros((txtnum, binumber))
# histacc = np.zeros((txtnum, binumber))

for i in range(txtnum):
    histct[i], bin_edges = np.histogram(arr[i, :], bins=binumber, range=(0, remax), density=True)