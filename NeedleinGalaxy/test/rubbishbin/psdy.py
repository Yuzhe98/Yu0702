#!/usr/bin/env python
import numpy
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import window
import expmovmed
from sklearn.preprocessing import normalize

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
        return numpy.log(2)
    ans = 1
    for i in range(1, int((n - 1) / 2 + 1)):
        ans += 1.0 / (2*i + 1) - 1.0 / (2*i)
    return ans

def normalize(a):
    return a/numpy.sum(a)

def excess_power(
    t_data=range(12),  # signal series in time domain
    samp_rate=512,  # sampling rate
    t_len=3600,  #length of measuring time
    seg_len=60,  # Length of each seg in seconds
    seg_strd=30,  # overlap length, sgement stride in seconds
    start_time=0,
    end_time = None,

    wtype='hanning0.5',  # Whitening type, can tukey or hann
    welchnps=512,

    avg_method='median',
    avg_opt=True,
    EMAwin=200,
    EMAfcor=0.4,
    make_plot=True,  # Condition to produce plots
    colormap_name='plasma',
    colorbar_log=True,
    station_name='station_name',  # Station name
    Freq_log=False,
    retopt=False,  #choose whether to output
    store_path='C:\\',
    store_opt_f_whited=False,
    store_opt_f_AC=False,
    store_opt_t_A=True,
    verbose=False,  # Print details
    prpr=False):


    #section to check the input
    #to be writeen!
    if end_time==None: end_time=t_len
    if seg_len<=seg_strd: raise ValueError("seg_len>=seg_strd")
    # to be writeen!

    series_len = int(t_len * samp_rate)
    seg_quantity=int((t_len - seg_len) / (seg_len-seg_strd))#int((t_len - seg_len + seg_strd) / seg_strd)

    seg_lenpt=int(seg_len * samp_rate)
    if verbose: print('seg_lenpt=', seg_lenpt)

    seg_lenpt_half=int((seg_lenpt + 1) / 2)
    if verbose: print('seg_lenpt_half=',seg_lenpt_half)
    seg_strdpt = int(seg_strd * samp_rate)
    # numpy.savetxt('C:\E\Yu0702\\,t_combined)
    data_reshape = numpy.zeros((seg_quantity,seg_lenpt), dtype=float)
    for i in range(seg_quantity):
        data_reshape[i] = t_data[i*seg_strdpt:i*seg_strdpt+seg_lenpt]

    #del t_data

    if wtype=='hanning0.5':
        w=window.hanning(seg_lenpt)
    else:
        raise ValueError('wtype!=hanning0.5')

    #f = fft(data_reshape*w)[:,0:seg_lenpt_half]#half of the whole series

    #f_welch = 2*numpy.abs(f)
    #f_welch[:,0] /= 2
    #f_welch[:,-1] /= 2
    #f_welch *= samp_rate / ((w*w)**0.5).sum()
    tf,f_welch=signal.welch(data_reshape,samp_rate,nperseg=welchnps)

    del data_reshape
    f_welch**=2

    aband=0
    f_avg=numpy.ones(\
        (\
            len(f_welch),len(f_welch[0])\
            ),dtype=float)
    if avg_method == 'expmovmed':
        f_avg,aband= expmovmed.ExpMovMed(f_welch,
                                         seg_len = seg_len,
                                         seg_str = seg_strd,
                                         pre_len = EMAwin,
                                         Efactor = EMAfcor)
    elif avg_method == 'mean':
        f_avg = numpy.mean(f_welch, axis=0)
    elif avg_method == 'median':
        f_avg = numpy.median(f_welch, axis=0) / median_bias(seg_quantity)
    elif avg_method == 'median-mean':
        odd_f_abs = f_welch[::2]
        even_f_abs = f_welch[1::2]
        odd_f_abs_avg = numpy.median(odd_f_abs, axis=0) / \
            median_bias(len(odd_f_abs))
        even_f_abs_avg = numpy.median(even_f_abs, axis=0) / \
            median_bias(len(even_f_abs ))
        f_avg = (odd_f_abs_avg + even_f_abs_avg) / 2.0

    if avg_opt:
        f_avged=f_welch/f_avg
    else:
        f_avged = f_welch
    if aband>0:
        f_welch=f_welch[aband:]
        f_avged = f_avged[aband:]
    #
    #f_welch=normalize(f_welch)
    #f_welch/=numpy.max(f_welch)
    f_welch_t=numpy.transpose(f_welch)
    #print('welch',len(f_welch_t),len(f_welch_t[0]))
    #f_avged/=f_avged.sum()
    f_avged_t=numpy.transpose(f_avged)
    #print('avged', len(f_avged_t), len(f_avged_t[0]))
    if store_opt_f_whited:
        numpy.savetxt(store_path+ "whitened" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + '.txt',f_welch)

    if make_plot:
        # time series
        time_input= numpy.linspace(start=start_time, stop=start_time+t_len, num=t_len * samp_rate, dtype=float)
        time = numpy.linspace(start=start_time+0.5*seg_strd, stop=start_time+0.5*seg_strd++t_len, num=seg_quantity, endpoint=False, dtype=float)
        if aband > 0:
            time = time[aband:]
        #freq = numpy.linspace(start=0.0, stop=samp_rate/2.0, num=seg_lenpt_half, endpoint=True, dtype=float)

        fig = plt.figure()#figsize=(7, 9)
        gs = gridspec.GridSpec(nrows=3, ncols=1)
        input_ax = fig.add_subplot(gs[0, 0])
        spec_ax=fig.add_subplot(gs[1,0])
        spec_w_ax=fig.add_subplot(gs[2,0])
        #spec_ax=fig.scatter(x, y, s=area2, marker='o', c=c)

        input_ax.plot(time_input, t_data)
        input_ax.set_xlabel('Time [s]')
        input_ax.set_ylabel('Amplitude')


        #print('var=',numpy.var(f_welch_t))
        #print('welch min=',f_welch_t.min())
        ''' '''
        if colorbar_log:
            c=spec_ax.pcolormesh(time, tf, f_welch_t, \
                                 norm=LogNorm(vmin=f_welch_t.min(),vmax=f_welch_t.max()), cmap=colormap_name)
            c_w = spec_w_ax.pcolormesh(time, tf, f_avged_t, \
                                   norm=LogNorm(vmin=f_avged_t.min(), vmax=f_avged_t.max()), cmap=colormap_name)
        else:
            c = spec_ax.pcolormesh(time, tf, f_welch_t,\
                                   vmin=f_welch_t.min(),vmax=f_welch_t.max(),cmap=colormap_name)
            c_w = spec_w_ax.pcolormesh(time, tf, f_avged_t, \
                                   vmin=f_avged_t.min(), vmax=f_avged_t.max(), cmap=colormap_name)

        #c=spec_ax.pcolormesh(time, freq, f_forfig,norm=LogNorm(vmin=f_forfig.min(),vmax=f_forfig.max()), cmap=colormap_name)
        #time_scatter=(time * numpy.ones((len(freq),len(time)))).flatten()
        #freq_scatter=numpy.transpose(freq * numpy.ones((len(time),len(freq)))).flatten()
        #c = spec_ax.scatter(x=time_scatter,y=freq, s=f_forfig,marker='o', cmap='plasma', alpha=0.75)

        spec_ax.set_title('not whitened', fontsize=20)
        spec_ax.set_xlabel('Time [sec]', fontsize=18)
        spec_ax.set_ylabel('Frequency [Hz]', fontsize=18)
        fig.colorbar(c, ax=spec_ax)

        spec_w_ax.set_title('whitened', fontsize=20)
        spec_w_ax.set_xlabel('Time [sec]', fontsize=18)
        spec_w_ax.set_ylabel('Frequency [Hz]', fontsize=18)
        fig.colorbar(c_w, ax=spec_w_ax)

        if Freq_log:
            spec_ax.set_yscale('log')
            spec_ax.set_ylim(1/seg_len, samp_rate / 2.)
            spec_w_ax.set_yscale('log')
            spec_w_ax.set_ylim(1/seg_len, samp_rate / 2.)
        fig.tight_layout()
        plt.show()

        fig = plt.figure()
        gs = gridspec.GridSpec(nrows=6, ncols=1)
        oned0_ax = fig.add_subplot(gs[0, 0])
        oned1_ax = fig.add_subplot(gs[1, 0])
        oned2_ax = fig.add_subplot(gs[2, 0])
        oned3_ax = fig.add_subplot(gs[3, 0])
        oned4_ax = fig.add_subplot(gs[4, 0])
        oned5_ax = fig.add_subplot(gs[5, 0])
        n = (1800) / (seg_len - seg_strd)-aband
        n = int(n)
        oned0_ax.plot(tf, f_avged[n - 2])
        oned1_ax.plot(tf, f_avged[n - 1])
        oned2_ax.plot(tf, f_avged[n])
        oned3_ax.plot(tf, f_avged[n + 1])
        oned4_ax.plot(tf, f_avged[n + 2])
        oned5_ax.plot(tf, f_avged[n + 3])
        oned5_ax.set_xlabel('freq [Hz]')
        fig.tight_layout()
        plt.show()



    if retopt:
        t_loc=1800.5
        bef=int(t_loc/seg_strd)
        if bef*seg_strd<=t_loc and (bef+1)*seg_strd>=t_loc:
            seg_loc=bef
        else:
            return 23333333
        axis0sum=numpy.sum(f_avged, axis=1)

        threshold=max(axis0sum[seg_loc - aband - 2:seg_loc - aband+1])
        count=[x for x in axis0sum if x>=threshold]
        r=(len(count)-1.0)/(1.0*len(axis0sum))
        ''''''
        print('not avged var/avg=', numpy.var(f_welch)/numpy.average(f_welch)**2)
        print('avged var/avg=',numpy.var(f_avged)/numpy.average(f_avged)**2)
        print(axis0sum[seg_loc - aband - 3])
        print(axis0sum[seg_loc - aband - 2])
        print(axis0sum[seg_loc - aband-1])
        print(axis0sum[seg_loc-aband])
        print(axis0sum[seg_loc - aband+1])
        print(axis0sum[seg_loc - aband + 2])
        print(axis0sum[seg_loc - aband + 3])
        print('threshold=',threshold)
        print('events num=',len(count),' in ',len(axis0sum))
        print('r=',r)
        #print('\n')
        return r
    return 0




'''
!!!dont delete!!!
hist_ax=fig.add_subplot(gs[0,1])
hist_ax.hist(f_forfig.flatten(), bins=500)
'''