import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec

x=np.exp(2j * np.pi * np.arange(8) / 8)
y=np.fft.fft(x,norm=None)
print(np.sum(abs(x)**2))
print(np.sum(abs(y)**2))