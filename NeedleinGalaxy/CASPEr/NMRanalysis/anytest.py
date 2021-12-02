import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec

t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()