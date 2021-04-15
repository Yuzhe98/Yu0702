import numpy as np
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
import epa
import f2t
'''
testdata=3.*np.random.randn(1,512*2)+10
testdata+=f2t.Lorentzian(
    samp_rate=512,  # sampling rate. Default to 512 Hz
    t_len=2,  # signal time length
    setend=False,

    amp=1.0,  # amplitude of the lorentzian peak
    pkloc=5,  # the location of the peak
    FWHM=0.09,  # Full Width at Half Maximum

    store_opt=False,  # condition to store data. Default to False
    store_path='C:\E',  # store path
    prefix='201907',  # prefix for name of stored document
    suffix='',  # suffix for name of stored document
    ret_opt=True  # condition to return something. Default to True
)


epa.ExpMovAvg(
        data=3.3*np.random.randn(40,97)+10,  # data to be whitened
        seg_len = 60,  # segment length
        str_len = 30,  # stride length
        pre_len = 100,  # the length of data series used to calculate average value
        Efactor = 0.4  # lamda in Exp function
)
'''

epa.ExPow(
    t_data=233+233*np.random.rand(int(512*3600)),  # signal series in time domain
    samp_rate=512,  # sampling rate. Default to 512 Hz, which is common for GNOME stations
    t_len=3600,  # length of measuring time. Default to 1 hour
    seg_len=60,  # Length of each seg in seconds. Default to 1 min
    str_len=30,  # overlap length, segement stride in seconds. Default to 0.5 min
    start_time=0,  # start time shown in the final figure
    end_time = None,  # end time shown in the final figure
    welchnps=512,  # data point number per segment in welch method. Plz refer to sicpy->signal->welch and welch method
    f_lpass=1,  # frequency low pass for keeping part of welch result. Ususally we don't care 0 Hz. Default to 1 Hz
                # since we usually do not care about the 0 Hz component
    f_hpass=256,  # frequency high pass for keeping part of welch result. Default to 256 Hz.
    avg_opt=True,  # Condition to whiten the data. Default to True
    avg_method='expmovavg',  # average method for whitening.
                             # Default to Exponentially Moving Average (EMA). Plz refer to wiki
    EMAwin = 200,  # EMA window length. Default to 200 sec
    EMAfcor=0.4,  # EMA factor. Default to 0.4
    make_plot=True,  # Condition to produce plots
    colormap_name='plasma',  # option of color map. Default to 'plasma'
    colorbar_log=False,  # condition to show color bar in log scale. Default to False
    station_name='station_name',  # station name to show in the figure
    freq_log=False,  # condition to show freq band in log scale. Default to False
    ret_opt=False,  # condition to return something
    store_path='C:\\',  # the path for storing data
    store_opt_f_whited=False,  # condition to store whitened data
    # you can add more here to store interested data
    verbose_opt=False,  # condition to print processing details
    prpr=False # parameter of no use
)
