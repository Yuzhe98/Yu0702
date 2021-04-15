import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def median_bias(n):
    """Calculate the bias of the median average PSD computed from `n` segments.

    Parameters
    ----------
    n : int
        Number of segments used in PSD estimation.

    Returns
    -------
    ans : float
        Calculated bias.

    Raises
    ------
    ValueError
        For non-integer or non-positive `n`.

    Notes
    -----
    See arXiv:gr-qc/0509116 appendix B for details.
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if n >= 1000:
        return np.log(2)
    ans = 1
    for i in range(1, int((n - 1) / 2 + 1)):
        ans += 1.0 / (2*i + 1) - 1.0 / (2*i)
    return ans

def ExpMovAvg(
        data=3.3*np.random.randn(40,97)+10,  # data to be whitened
        seg_len = 60,  # segment length
        str_len = 30,  # stride length
        pre_len = 100,  # the length of data series used to calculate average value
        Efactor = 0.4  # lamda in Exp function
):
    """
    Exponentially Moving Average
    You may also find it described as Running Median (Exponential), which could be confusing
    See Exponentially Moving Average (EMA) on wiki for details
    """
    t_delta = seg_len - str_len
    alpha = np.exp(-1.0 * Efactor * t_delta)
    pre_segnum = int(pre_len/t_delta)  # the number of segments used to calculate average
    # here I decide to abandon the first part of data used to calculate average finally
    # since we are not able to whiten this part
    if pre_segnum<=0:  # check the validity of inputs
        raise ValueError('pre_segnum<=0')
    if (pre_segnum>=1000):  # limit pre_segnum within 1000 in case out of our processing ability
        raise ValueError('pre_segnum>=1000')
    y_len = len(data)
    x_len = len(data[0])
    wet_sum = (1 - np.power(alpha, pre_segnum)) / (1 - alpha)  # sum of all weights
    i=0
    loop_times = pre_segnum
    ai = (alpha) ** (i) / wet_sum
    EMA_array = np.ones((y_len, x_len), dtype=float)  # initialization of average 2D array
    EMA_array[pre_segnum:] = ai * data[pre_segnum - i - 1:y_len - i - 1]
    for i in np.arange(start=1, stop=loop_times, step=1):
        ai = (alpha) ** (i)/wet_sum
        EMA_array[pre_segnum:] += ai * data[pre_segnum - i - 1:y_len - i - 1]

    return EMA_array,pre_segnum  # return the average array and
                                # number of abandoned segments

def ExPow(
    t_data=range(12),  # signal series in time domain
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
):
    """
    Excess Power Analysis
    Here I name it as 'ExPow'. I wish you would never understand what is ex's power.
    Yu
    """
    # section to check the validity of input
    # to be perfected!
    if end_time==None: end_time=t_len

    if seg_len<=str_len: raise ValueError("seg_len>=seg_strd")#stride should be greater than segment

    if f_lpass<0:raise ValueError('f_lpass<0')
    if f_lpass>f_hpass: raise ValueError('f_lpass>f_hpass')
    if f_hpass>samp_rate/2: raise ValueError('f_hpass>samp_rate/2')# freq high pass should be smaller than freq limit,
    # which equals to hald of sampling rate

    f_resolution = samp_rate / welchnps  # calculate the freq resolution
    H = int(f_hpass / f_resolution)
    L = int(f_lpass / f_resolution)
    if L > H: raise ValueError('L > H')
    # more checks to be added here!

    #convert 1D data in time domain into 2D array for applying welch method later
    seg_quantity=int((t_len - seg_len) / (seg_len - str_len))  # calculate the number of segments
    if verbose_opt: print('seg_quantity=', seg_quantity)
    seg_num=int(seg_len * samp_rate)  # number of data points in one segment
    if verbose_opt: print('seg_num=', seg_num)
    strd_num = int(str_len * samp_rate)  # number of data points in one stride
    if verbose_opt: print('strd_num=', strd_num)
    data_reshape = np.zeros((seg_quantity,seg_num), dtype=float)
    for i in range(seg_quantity):
        data_reshape[i] = t_data[i*strd_num:i*strd_num+seg_num]
    # apply welch method
    tf,f_welch = signal.welch(data_reshape,samp_rate,nperseg=welchnps)  # tf is freq band,
                                                            # f_welch is the root of power density
    del data_reshape  # delet the reshaped 2D array
    f_welch=f_welch[:,L:H+1]  # keep the interested freq band
    tf = tf[L:H+1]  # keep the interested freq band
    f_welch**=2  # actually it doesn't matter whether to use the square of welch result
    # whitening procedure
    aband = 0  # sometimes we have to abandon the beginning of the data. 'aband' records the number of segments we
            #have to abandon
    f_avg=np.ones(\
        (\
            len(f_welch),len(f_welch[0])\
            ),dtype=float)  # initialize a 2D array to record the averaged data
    if avg_opt:
        if avg_method == 'expmovavg':  # Exponentially Moving Average (EMA)
            f_avg, aband = ExpMovAvg(f_welch,
                                     seg_len=seg_len,
                                     str_len=str_len,
                                     pre_len=EMAwin,
                                     Efactor=EMAfcor)
        elif avg_method == 'mean':  # mean method
            f_avg = np.mean(f_welch, axis=0)
        elif avg_method == 'median':  # median method
            f_avg = np.median(f_welch, axis=0) / median_bias(seg_quantity)
        elif avg_method == 'median-mean':  # median-mean method
            odd_f_abs = f_welch[::2]
            even_f_abs = f_welch[1::2]
            odd_f_abs_avg = np.median(odd_f_abs, axis=0) / \
                            median_bias(len(odd_f_abs))
            even_f_abs_avg = np.median(even_f_abs, axis=0) / \
                             median_bias(len(even_f_abs))
            f_avg = (odd_f_abs_avg + even_f_abs_avg) / 2.0

        f_wtd=f_welch/f_avg  # to whiten data
    else:
        f_wtd = f_welch  # not to whiten data

    if aband>0:  # to abandon not-whitened data
        f_welch=f_welch[aband:]
        f_wtd = f_wtd[aband:]

    f_welch_t=np.transpose(f_welch)  #
    f_wtd_t=np.transpose(f_wtd)
    # in f_welch and f_wtd, each array records the signals in one certain time segment over freq bands
    # in f_welch_t and f_wtd_t, each array records the signals in one certain freq band over measuring time

    # to store interested data
    if store_opt_f_whited:
        np.savetxt(store_path+ "whitened" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + '.txt',f_wtd)
    # to make plot
    # you can make your own interested plot
    if make_plot:
        # time series
        time_input= np.linspace(start=start_time, stop=start_time+t_len, num=t_len * samp_rate, dtype=float)
        time = np.linspace(start=start_time + 0.5 * str_len, stop=start_time + 0.5 * str_len + +t_len, num=seg_quantity, endpoint=False, dtype=float)
        if aband > 0:
            time = time[aband:]

        fig = plt.figure()
        gs = gridspec.GridSpec(nrows=3, ncols=1)
        input_ax = fig.add_subplot(gs[0, 0])
        spec_ax=fig.add_subplot(gs[1,0])
        spec_w_ax=fig.add_subplot(gs[2,0])

        input_ax.plot(time_input, t_data)
        input_ax.set_xlabel('Time [s]', fontsize=16)
        input_ax.set_ylabel('Amplitude', fontsize=16)

        if colorbar_log:
            c=spec_ax.pcolormesh(time, tf, f_welch_t, \
                                 norm=LogNorm(vmin=f_welch_t.min(),vmax=f_welch_t.max()), cmap=colormap_name)
            c_w = spec_w_ax.pcolormesh(time, tf, f_wtd_t, \
                                   norm=LogNorm(vmin=f_wtd_t.min(), vmax=f_wtd_t.max()), cmap=colormap_name)
        else:
            c = spec_ax.pcolormesh(time, tf, f_welch_t,\
                                   vmin=f_welch_t.min(),vmax=f_welch_t.max(),cmap=colormap_name)
            c_w = spec_w_ax.pcolormesh(time, tf, f_wtd_t, \
                                   vmin=f_wtd_t.min(), vmax=f_wtd_t.max(), cmap=colormap_name)

        spec_ax.set_title('not whitened', fontsize=18)
        spec_ax.set_xlabel('Time [sec]', fontsize=16)
        spec_ax.set_ylabel('Frequency [Hz]', fontsize=16)
        fig.colorbar(c, ax=spec_ax)
        spec_w_ax.set_title('whitened', fontsize=18)
        spec_w_ax.set_xlabel('Time [sec]', fontsize=16)
        spec_w_ax.set_ylabel('Frequency [Hz]', fontsize=16)
        fig.colorbar(c_w, ax=spec_w_ax)
        if freq_log:
            spec_ax.set_yscale('log')
            spec_ax.set_ylim(f_lpass,f_hpass)
            spec_w_ax.set_yscale('log')
            spec_w_ax.set_ylim(f_lpass,f_hpass)
        fig.tight_layout()
        plt.show()
    # you can return what you are interested in
    if ret_opt:
        return 0
    return 0
