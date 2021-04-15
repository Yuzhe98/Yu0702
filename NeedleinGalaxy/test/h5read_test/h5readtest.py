import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import h5read
'''
beijing01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\\beijing01\\2019\\01\\26'
berkeley01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\\berkeley01\\2019\\01\\26'
daejeon01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\daejeon01\\2019\\01\\26'
hayward01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\hayward01\\2019\\01\\26'
hefei01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\hefei01\\2019\\01\\26'
lewisburg01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\lewisburg01\\2019\\01\\26'
mainz01dir = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\mainz01\\2019\\01\\26'

saveway = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\202001testdata\savetxt'
'''
filepath = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\201901\\24\\Uploaded'
filepath2= 'H:\GNOME_data\\202001testdata\\beijing01\\2019\\01\\24'
filepath3 = 'H:\GNOME_data\\202001testdata\mainz01\\2019\\01\\24'
'''
h5read.h5process(
        station_name='beijing01',  # Hefei01
        curdir=filepath2,  # current directory
        samp_rate=512,
        year=2019,
        month=1,
        day=24,  # np.array([21,31]),
        start=1315,  # 1309
        end=1745,  # 1922
        sec=56,
        suffix='.hdf5',
        avg_method='tensec',
        fft_plot_opt=True,
        plot_opt=False,

        store_opt=False
        #storepath=saveway
)'''
h5read.h5process(
        station_name='Hefei01',  # Hefei01
        curdir=filepath,  # current directory
        samp_rate=512,
        year=2019,
        month=1,
        day=24,  # np.array([21,31]),
        start=1309,  # 1309
        end=1922,  # 1922
        sec=11,
        suffix='.H5',
        avg_method='tensec',
        fft_plot_opt=True,
        plot_opt=False,

        store_opt=False
        #storepath=saveway
)
'''
beijing01name = "Station_beijing01_20190126_0000_to_1200_FFT_favg_welch"
berkeley01name = "Station_berkeley01_20190126_1100_to_2359_FFT_favg_welch"
#daejeon01name = ""
hayward01name = "Station_hayward01_20190126_0000_to_2359_FFT_favg_welch"
hefei01name = "Station_hefei01_20190126_0000_to_1100_FFT_favg_welch"

#fnamearray=np.array([beijing01name,berkeley01name,hayward01name,hefei01name])


MultiDisplay(
        directory = saveway,
        fname = fnamearray,
        verbose = True
)'''



