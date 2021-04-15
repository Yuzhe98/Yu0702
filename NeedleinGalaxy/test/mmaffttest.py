import numpy as np
from scipy import signal
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
a = np.loadtxt("C:\E\\USTC\MagLab\equ\mma files\\0409\\test23_sr13531to10.txt")
#a=np.transpose(a)
a=a.flatten()
samp_rate=13531/10.

tf, f_welch=signal.welch(a, samp_rate,nperseg=len(a))
print(a.shape)
plt.figure()
plt.plot(a)  # linestyle=':', marker='o',
plt.xlabel('time')
plt.ylabel('Amplitude [a.u.]')
#plt.title('Station: ' + station_name + ' ' +
#          'Y/M/D: %04d/%02d/%02d Hour&Minute: %04d to %04d'
#          % (year, month, day, ta[0], ta[-1]) + '\nFFT')
plt.show()

plt.figure()
plt.plot(tf, f_welch)  # linestyle=':', marker='o',
plt.xlabel('Freq [Hz]')
plt.ylabel('Amplitude [a.u.]')
#plt.title('Station: ' + station_name + ' ' +
#          'Y/M/D: %04d/%02d/%02d Hour&Minute: %04d to %04d'
#          % (year, month, day, ta[0], ta[-1]) + '\nFFT')
plt.show()
print(tf)