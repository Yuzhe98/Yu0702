import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import ngenerator
import psdy
from scipy import signal
for i in range(8):
    fwhm=10*(0.1**i)
    Lz=ngenerator.Lorentzian(
    starttime=0.0,
    samp_rate=512,  # sampling rate
    t_len=100,
    setend=False,

    amp=1.0,
    pkloc=50,
    FWHM=fwhm)
    print('FWHM=',fwhm)
    print('Freq=',1/fwhm)
    tf,p=signal.welch(Lz,512,nperseg=512)
    plt.plot(tf,p)
    plt.show()