import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import scipy.optimize
import matplotlib.gridspec as gridspec
def Lorentzian(x, center, gamma, A, offset):
    return 0.5 * gamma * A / (np.pi * ((x - center) ** 2 + (0.5 * gamma) ** 2))  # offset +
    #return offset + A / (((x-center)**2/(gamma)**2) + 1)


def LIAFilterPSD(
        frequency = None,
        taun = 2.5486e-5,
        order = 8
):
    '''
    based on 6.4.1. Discrete-Time RC Filter https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf
    '''
    return np.abs(1 / (1 + 2 * np.pi * 1j * frequency * taun) ** order) ** 2
def stdOscpPSD(
        data,
        samprate = 1e3,  # in Hz
        dwelltime = 1,
        attenuation = 0,  # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
        window = 'rectangle',  # Hanning, Hamming, Blackman
        DTRCfilter='off',
        DTRCfilter_Tc=2.5486e-5,
        DTRCfilter_order=8,
        verbose=False,

):
    '''
    Based on https://holometer.fnal.gov/GH_FFT.pdf
    '''
    #print('samprate ',samprate)
    if window == 'rectangle':
        windowfunction = np.ones(len(data))
    elif window == 'Hanning' or window == 'hanning' or window == 'Han' or window == 'han':
        windowfunction = np.hanning(len(data))
    elif window == 'Hamming' or window == 'hamming':
        windowfunction = np.hamming(len(data))
    elif window == 'Blackman' or window == 'blackman':
        windowfunction = np.blackman(len(data))
    else:
        raise ValueError('Window function not found')
    if verbose:
        print('shape of window function ', windowfunction.shape)
    #S1 = np.sum(windowfunction)
    S2 = np.sum(windowfunction**2)
    if verbose:
        print('stdLIAPSD S2 = %g'%S2)
    frequencies = np.fft.fftfreq(len(data), d=dwelltime)  # Set d to dwell time in s
    #print(frequencies.shape)
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter=='on':
        filtercomp /=LIAFilterPSD(
            frequency = frequencies,
            taun = DTRCfilter_Tc,
            order = DTRCfilter_order)

    FFT = np.fft.fft(a=data * windowfunction, norm=None)
    PSD = 10.0**(attenuation/10.) * 2.0 * np.abs(FFT) ** 2 / (S2 * samprate) * filtercomp
    if verbose:
        print('shape of PSD ', PSD.shape)
        #print('FFT ', FFT[:100])
    return np.sort(frequencies), PSD[np.argsort(frequencies)]


filepath='D:\\Mainz\\CASPEr\\20211130 Keadata\\Noise Analysis\\data\\'
filename_arr=np.array(['test_dc_1','test_dc_2','test_dc_3','test_dc_4',
                       'test_DCAMPoff_1M','test_DCAMPoff_10M',
                       'test_DCAMPon_1M','test_DCAMPon_10M',
                       'test_DCAMPonFGoff_1M','test_DCAMPonFGoff_10M',
                       'test_DCAMPonFGon_1M','test_DCAMPonFGon_10M'
                       ])
savename_arr=np.array(['test_dc_1','test_dc_2','test_dc_3','test_dc_4',
                       'DC Amp power off_1M','DC Amp power off_10M',
                       'DC Amp power on_1M','DC Amp power on_10M',
                       'FunctionGenerator output off__1M','FunctionGenerator output off_10M',
                        'FunctionGenerator output on_1M','FunctionGenerator output on_10M'
                        ])
for i in range(len(filename_arr)):
    #testdata = np.genfromtxt(filepath + 'test_464mv_1'+'.csv', delimiter=",", skip_header=2, unpack=True)
    #print(testdata.shape)
    time_arr, data = np.loadtxt(filepath + filename_arr[i] + '.csv', delimiter=",", skiprows=2, unpack=True)
    data = data / 10. / 0.1 * 21.92 / 2.37
    #print('data  ', data)
    print('data.shape = ',data.shape)
    print(len(data))
    #print('data = ', data[:100])
    #print(data[0].shape)
    #print(data[0, :].shape)
    #print(data[:, 0].shape)
    print('abs(time_arr[0]-time_arr[1]) = ',abs(time_arr[0]-time_arr[1]),' s')
    #print('samprate=1 / abs(time_arr[0]-time_arr[1]) = ', 1. / abs(time_arr[0]-time_arr[1]))

    #print('fft nan? ',np.fft.fft(a=data, norm=None)[:100])

    frequencies, spectrum = stdOscpPSD(data=data, dwelltime = 1000*abs(time_arr[0]-time_arr[1]), verbose=True, attenuation=0)  # [2:len(data)-10]
    plt.rcParams['font.size'] = 14
    fig = plt.figure(figsize=(12, 12 * 9 / 21))  #
    gs = gridspec.GridSpec(nrows=1, ncols=2)  #
    left_spc = 0.055
    bottom_spc = .133
    right_spc = 0.99
    top_spc = 0.912
    xgrid_spc = .271
    ygrid_spc = .1
    fig.subplots_adjust(left=left_spc, right=right_spc, top=top_spc, bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    signal_ax = fig.add_subplot(gs[0, 0])
    signal_ax.locator_params(nbins=6, axis='x')
    signal_ax.plot(time_arr, data, label='standard deviation = %.2g V'%np.std(data),c='tab:blue')
    signal_ax.set_xlabel('time / s')
    signal_ax.set_ylabel('Bz coil voltage / V')
    signal_ax.set_xlim(-0.027, 0.027)
    signal_ax.set_ylim(-3.5, 27)
    #signal_ax.set_title('standard deviation = %.2g V'%np.std(data))
    signal_ax.legend(loc='upper right')

    freqscale = 1
    spec_ax = fig.add_subplot(gs[0, 1])
    spec_ax.locator_params(nbins=6, axis='x')

    fstart = np.argmin(abs(frequencies - 1))
    #spec_ax.plot(spectrum, label='')
    spec_ax.plot(freqscale*frequencies[fstart:],spectrum[fstart:],c='tab:green')
    spec_ax.plot(freqscale * frequencies[fstart+15000:], spectrum[fstart+15000:],
                 label='standard deviation = %.2g$\\times 10^{-4}V^2/Hz$' % (1e4*np.std(spectrum[fstart+15000:])),c='tab:orange',alpha=1)
    spec_ax.set_xlabel('absolute frequency / kHz')
    spec_ax.set_ylabel('Amplitude / $V^2/Hz$')
    #spec_ax.set_title('standard deviation = %.2g$\\times 10^{-4}V^2/Hz$' % (1e4*np.std(spectrum[fstart:])))
    spec_ax.legend(loc='upper right')  #
    #spec_ax.set_xlim(0.1, )
    spec_ax.set_ylim(-0.001, 0.03)
    #spec_ax.set_yscale("log")
    #fig.suptitle(filepath + filename_arr[i] + '.csv')
    fig.suptitle('sampling rate %g MHz'%(1e-6/abs(time_arr[0]-time_arr[1])))
    plt.grid()
    plt.savefig(filepath + savename_arr[i] + '.png')
    #plt.show()
    plt.close()
    del time_arr, data, fig, gs

'''
def OscpPSDplot(
    fpath='D:\\Mainz\\CASPEr\\20211130 Keadata\\20211130\\',
        fname = 'test5.csv',
):
    
'''