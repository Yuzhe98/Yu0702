import numpy as np
import matplotlib.pyplot as plt
import NMRanalysis as NMRa
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec

NMRa.pulseNMRplot(
        file="D:\\Mainz\\CASPEr\\20211208 SQUID shim\\stream_088/stream_00000.h5",
        attenuation=6,  # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
        window='rectangle',  # Hanning, Hamming, Blackman
        DTRCfilter='on',

        dfreq=None,  # in Hz
        samprate=6696,  # in Hz
        pulselength=450e-6,
        acqdelay=600e-6,
        acqtime=100e-3,
        showtimedomain=True,
        showacqdata=True,
        showfreqdomain=True,
        showstd=False,
        spectype='FluxPSD',
        Mf=1 / (44.12e-6),  # feedback sensitivity
        Rf=10e3,  # in Ohm
        frequnit='MHz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
        ampunit='muPhi',  # 'V', 'muV', 'muPhi'
        ampscale='linear',
        singlePSD_arr=[1,2],
        stddev_range=[35e3, 37e3],
        verbose=True
    )