import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
dfreq = 2.0893976
freq0 =  np.loadtxt("D:\\Mainz\\CASPEr\\Patras 2021\\pyplot\\119\\frequencies.txt")
spec0 = np.loadtxt("D:\\Mainz\\CASPEr\\Patras 2021\\pyplot\\119\\spectrum.txt")
freq1 =  np.loadtxt("D:\\Mainz\\CASPEr\\Patras 2021\\pyplot\\frequencies.txt")
spec1 = np.loadtxt("D:\\Mainz\\CASPEr\\Patras 2021\\pyplot\\spectrum.txt")

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(freq0[0:len(freq0)//2]-1.759,spec0[0:len(freq0)//2],c='grey')  #/ (200.0 + 200.0 * i)
ax0.plot(freq0[len(freq0)//2+1:]-1.759,spec0[len(freq0)//2+1:],c='grey')  # '#1f77b4'
ax0.set_xlabel('Frequency [kHz]')  #, fontsize=16
ax0.set_ylabel('Amplitude [$\mu$V/kHz]')  #, fontsize=16
ax0.set_xlim(-2,2)
ax0.set_ylim(-.12,5)
#ax1.set_title('', fontsize=18)
#ax0.legend(loc='upper right')
#ax1 = fig.add_subplot(gs[0, 1])
#ax1.plot(freq1[0:len(freq1)//2]+0.832,spec1[0:len(freq1)//2],c='#1f77b4')  #/ (200.0 + 200.0 * i)
#ax1.plot(freq1[len(freq1)//2+1:]+0.832,spec1[len(freq1)//2+1:],c='#1f77b4')
#ax1.set_xlabel('Frequency [kHz]')  #, fontsize=16
#ax1.set_ylabel('Amplitude [$\mu$V/kHz]')  #, fontsize=16
#ax1.set_xlim(-4,4)
#ax1.set_ylim(-4,57.5)
plt.show()