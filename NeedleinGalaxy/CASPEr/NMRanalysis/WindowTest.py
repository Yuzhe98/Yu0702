import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.gridspec as gridspec
from numpy.random import Generator, PCG64
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.default_rng().normal(mu, sigma, 1000)

samp_rate = 5*2.178*10**6
nu = 2.178*10**6  # MHz
nu1 = 2.178*10**5  # MHz
T = 1/nu

A = 1
numT = 100
timeseries_0 = np.linspace(start=0, stop=numT * T, num=round(samp_rate * numT * T) + 1, endpoint=True)
timeseries_1 = np.linspace(start=0, stop=(numT - 0.25) * T, num=round(samp_rate * (numT - 0.25) * T) + 1, endpoint=True)

signal_t_0 = np.sin(2 * np.pi * nu * timeseries_0) + np.sin(2 * np.pi * nu/10 * timeseries_0) + np.sin(2 * np.pi * nu/100 * timeseries_0)
signal_t_0[200:300]+=10*np.sin(2 * np.pi * nu * timeseries_0[200:300])
signal_t_1 = np.sin(2 * np.pi * nu * timeseries_1)+np.sin(2 * np.pi * nu/10 * timeseries_1)+np.sin(2 * np.pi * nu/100 * timeseries_1)


print('1')
dwellt= 1/samp_rate  # Set dwell time in ms
spectrum_0 = np.fft.fft(signal_t_0)  # , norm='ortho'
spectrum_0 = np.abs(spectrum_0) * dwellt  #**2
frequencies_0 = -np.fft.fftfreq(len(spectrum_0), d=dwellt) # Set d to dwell time in ms
print('2')
# rectangle window, non-integer period
spectrum_1 = np.fft.fft(signal_t_1)  # , norm='ortho'
spectrum_1 = np.abs(spectrum_1) * dwellt  #**2
frequencies_1 = -np.fft.fftfreq(len(spectrum_1), d=dwellt) # Set d to dwell time in ms

# Han window, non-integer period
spectrum_2 = np.fft.fft(np.hanning(len(signal_t_0))*signal_t_0)  # , norm='ortho'
spectrum_2 = np.abs(spectrum_2) * dwellt  #**2
frequencies_2 = -np.fft.fftfreq(len(spectrum_2), d=dwellt) # Set d to dwell time in ms


fig = plt.figure()  # figsize=(10, 10)
gs = gridspec.GridSpec(nrows=3, ncols=2)  #
signal_0_ax = fig.add_subplot(gs[0, 0])
signal_0_ax.scatter(timeseries_0, signal_t_0)

spec_0_ax = fig.add_subplot(gs[0, 1])
spec_0_ax.plot(frequencies_0, spectrum_0)
            #spec1_ax.set_ylabel('Amplitutude')
            #spec1_ax.set_xlabel('Freq [kHz]')
            #spec1_ax.set_title('')
#

#
signal_1_ax = fig.add_subplot(gs[1, 0])
signal_1_ax.scatter(timeseries_1, signal_t_1)
spec_1_ax = fig.add_subplot(gs[1, 1])
spec_1_ax.plot(frequencies_1, spectrum_1)
#
signal_2_ax = fig.add_subplot(gs[2, 0])
signal_2_ax.scatter(timeseries_0, np.hanning(len(signal_t_0)) * signal_t_0)
spec_2_ax = fig.add_subplot(gs[2, 1])
spec_2_ax.plot(frequencies_2, 2*spectrum_2)

plt.show()

