import numpy as np
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import matplotlib.pyplot as plt
import ngenerator

T=4
ngenerator.ftgenerator(
    samp_rate=512,  # sampling rate
    T=T,  # length of measuring time

    BG0_amp=10,

    BG1_amp=50,
    BG1_shape=-0.14,

    BG2_amp=0.00,
    BG2_loc=50,
    BG2_shape=0.009,

    Gz_sigma=1,
    Gz_mu=0,

    Lz0_amp=25.0,
    Lz0_pkloc=50,
    Lz0_FWHM=5,

    Lz1_amp=0.0,
    Lz1_pkloc=25,
    Lz1_FWHM=5,

    Lz2_amp=0.0,
    Lz2_pkloc=100,
    Lz2_FWHM=5,

    phase_distribution='Uniform',

    retopt=True,  # choose whether to output

    store_path="C:\E\Yu0702\\20190719\\",
    prefix='20190719',
    suffix=', 25Hz, 50Hz, 100Hz4',
    store_opt_f_amp=False,
    store_opt_f_AC=False,
    store_opt_t_A=True,
    make_plot=True,  # Condition to produce plots
    Freqlog=False,
    verbose=False,  # Print details
    prpr=False
)




'''
noisey.Lorentzian(
    starttime=0.0,
    samp_rate=512,  # sampling rate
    t_len=T,
    setend=False,

    amp=0.025,
    pkloc=0.5*T,
    FWHM=0.1**(i-1),

    store_opt=True,
    store_path="C:\E\Yu0702\\20190718\\",
    prefix='20190718',
    suffix='')
'''
'''
s=noisey.SineEDec(
    starttime=0.0,
    samp_rate=512,  # sampling rate
    t_len=T,
    setend=False,

    amp=1,
    Elambda=1,
    freq=50,
    phase0=0,
    startloc=0.5*T,

    store_opt=True,
    store_path=path,
    prefix='20190716',
    suffix=''
)
time = np.linspace(start=0, stop=3600, num=3600*512, endpoint=False, dtype=float)
plt.plot(time,s)
plt.show()
'''