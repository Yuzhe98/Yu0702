import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.gridspec as gridspec

rng = np.random.default_rng()

samp_rate = 20*2.178*10**6
nu = 10.178*10**6  # MHz
T = 1/nu

A = 1
numT = 5
timeseries0 = np.linspace(start=0, stop=numT * T, num=round(samp_rate * numT * T) + 1, endpoint=True)

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.default_rng().normal(mu, sigma, 1000)