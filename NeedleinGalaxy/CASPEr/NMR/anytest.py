import numpy as np
import matplotlib.pyplot as plt
import NMRanalysis as NMRa
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec

#file="D:\\Mainz\\CASPEr\\20211208 SQUID shim\\stream_"+str(139)+"/stream_00000.h5",
#file="D:\\Mainz\\CASPEr\\20211208 SQUID shim\\session_20211210_123550_000" + "/session_20211210_123550_00000.h5",
sw=2
if sw==1:
        NMRa.pulsedNMRplot(
                #file="D:\\Mainz\\CASPEr\\20211208 SQUID shim\\stream_"+str(139)+"/stream_00000.h5",
                file="D:\\Mainz\\CASPEr\\20211208 SQUID shim\\session_20211210_123550_000" + "/session_20211210_123550_00000.h5",
                loadstream_opt=2,
                attenuation=6,  # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
                window='rectangle',  # Hanning, Hamming, Blackman
                DTRCfilter='on',

                #dfreq=None,  # in Hz
                #samprate=13.39e3,  # in Hz
                pulselength=225e-6,
                acqdelay=600e-6,
                acqtime=100e-3,
                showtimedomain=True,
                showacqdata=True,
                showfreqdomain=True,
                showstd=False,
                fit_opt=True,
                spectype='FluxPSD',
                Mf=1 / (44.12e-6),  # feedback sensitivity
                Rf=10e3,  # in Ohm
                frequnit='Hz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
                ampunit='muPhi',  # 'V', 'muV', 'muPhi'
                ampscale='linear',
                singlePSD_arr=[],
                stddev_range=[1.34e6, 1.35e6],
                verbose=True
        )
if sw==2:
        NMRa.spinnoiseplot(
                file_arr=["D:\\Mainz\\CASPEr\\20211208 SQUID shim\\stream_"+str(153)+"/stream_00000.h5"],
                loadstream_opt=1,
                attenuation=6,
                # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
                window='rectangle',  # Hanning, Hamming, Blackman
                DTRCfilter='on',
                filter_TC=7.980055670486763e-06,
                filter_order=8,
                dfreq=0,  # in Hz
                samprate=13392.857421875,  # in Hz

                chunksize=500e-3,  # in second
                analysisrange=[120, -1],

                showtimedomain=False,
                showacqdata=False,
                showfreqdomain=True,
                showstd=False,

                fit_opt=False,
                spectype='FluxPSD',  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
                Mf=1 / (44.12e-6),
                # feedback sensitivity which can be found in the SQUID specifications. For the SQUID we usually use, M_f = 31 706 \phi_0 / A
                Rf=10e3,  # in Ohm
                frequnit='Hz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
                ampunit='muPhi',
                ampscale='log',  # 'log', 'linear'
                singlePSD_arr=[],
                stddev_range=[35e3, 37e3],

                left_spc=0.1,
                top_spc=1 - 0.1,
                right_spc=1 - .05,
                bottom_spc=.1,
                xgrid_spc=.3,
                ygrid_spc=.2,
                verbose=True
        )