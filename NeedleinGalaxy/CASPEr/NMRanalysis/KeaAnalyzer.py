import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec
import scipy.optimize

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


def stdLIAPSD(
        data_x,
        data_y,
        dataunit = 'V',  # 'muV', 'nV'
        samprate = 1e3,  # in Hz
        dfreq = 30e3,  # in Hz
        attenuation = 6,  # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
        window = 'rectangle',  # Hanning, Hamming, Blackman
        DTRCfilter='off',
        DTRCfilter_Tc=2.5486e-5,
        DTRCfilter_order=8,
        verbose=False,

):
    '''
    Based on https://holometer.fnal.gov/GH_FFT.pdf
    '''
    if len(data_x) != len(data_y):
        raise ValueError('len(data_x) != len(data_y)')

    if window == 'rectangle':
        windowfunction = np.ones(len(data_x))
    elif window == 'Hanning' or window == 'hanning' or window == 'Han' or window == 'han':
        windowfunction = np.hanning(len(data_x))
    elif window == 'Hamming' or window == 'hamming':
        windowfunction = np.hamming(len(data_x))
    elif window == 'Blackman' or window == 'blackman':
        windowfunction = np.blackman(len(data_x))
    else:
        raise ValueError('Window function not found')
    #S1 = np.sum(windowfunction)
    S2 = np.sum(windowfunction**2)
    if verbose:
        print('stdLIAPSD S2 = %g'%S2)
    frequencies = np.fft.fftfreq(len(data_x), d=1 / samprate)  # Set d to dwell time in s
    #print(frequencies.shape)
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter=='on':
        filtercomp /=LIAFilterPSD(
            frequency = frequencies,
            taun = DTRCfilter_Tc,
            order = DTRCfilter_order)

    FFT = np.fft.fft((data_x + 1j * data_y) * windowfunction, norm=None)
    PSD = 10.0**(attenuation/10.) * 1.0 * np.abs(FFT) ** 2 / (S2 * samprate) * filtercomp

    frequencies += dfreq
    return np.sort(frequencies), PSD[np.argsort(frequencies)]


filepath='D:\\Mainz\\CASPEr\\20211130 Keadata\\20211130\\'  # 37\\data.csv
#filepath='C:\\Users\\Public\\Downloads\\'  # 37\\data.csv
#folde_arr =  np.array([109,110,111,112])
folde_arr =  np.array([113,114,115,116])
numofscan_arr = np.array([10,20,30,40,50,60])
# numofscan = np.array([10,20,30,40,50,60])

for i in range(len(folde_arr)):

    acqfile = open(filepath + str(folde_arr[i]) + '\\acqu.par', 'r')
    acqpara = acqfile.read()
    acqfile.close()
    #for j in range(len(acqpara)):
    #    print(j,'    ',acqpara[j])

    dwelltime = 500.0e-6
    dmodfreq = 0.310664e6
    print(filepath + str(folde_arr[i])+'\\data.csv')
    data = np.loadtxt(filepath + str(folde_arr[i])+'\\data.csv', delimiter=",")

    #print(data.shape)
    #print(data[0].shape)
    #print(data[0, :].shape)
    #print(data[:, 0].shape)

    freq, spectrum = stdLIAPSD(data_x=data[:, 1], data_y=data[:, 2], samprate=1 / dwelltime, dfreq=dmodfreq)
    ar = [2*len(freq) // 5, len(freq) - 2*len(freq) // 5]

    #print(spectrum.shape)
    #print("abs(freq[0]-freq[1])) = %g " % (abs(freq[0] - freq[1])))
    intgeral = np.sum(spectrum[ar[0]:ar[1]]) * abs(freq[0] - freq[1])
    print("integral = %.4g uV^2" % (intgeral))
    print("sqrt root of integral = %.4g uV" % (np.sqrt(intgeral)))
    peakpara = (freq[np.argmax(spectrum[ar[0]:ar[1]])+ar[0]],50*2,np.amax(spectrum[ar[0]:ar[1]])*np.pi*50,0.0)  # center, gamma, A, offset
    #3.10683529e+05, 7.50595548e+01,   1.80273899e-04,    0.00000000e+00
    #print("peakpara = ", peakpara)

    fig = plt.figure(figsize=(8, 8*9/16))  #
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    freqscale = 1e-3
    spec_ax = fig.add_subplot(gs[0, 0])
    spec_ax.locator_params(nbins=10, axis='x')
    spec_ax.plot(freqscale*freq, spectrum)
    spec_ax.plot(freqscale*freq[ar[0]:ar[1]], spectrum[ar[0]:ar[1]], label='signal peak\nnumeric integral = %.4g uV^2 \nsqrt root of integral = %g uV'%(intgeral,np.sqrt(intgeral)))
    #popt, pcov = scipy.optimize.curve_fit(Lorentzian, freq[ar[0]:ar[1]], spectrum[ar[0]:ar[1]], peakpara)
    try:
        popt, pcov = scipy.optimize.curve_fit(Lorentzian, freq[ar[0]:ar[1]], spectrum[ar[0]:ar[1]], peakpara)
        fitcurve = Lorentzian(freq[ar[0]:ar[1]], popt[0], popt[1], popt[2], popt[3])
        intgeral_fit = np.sum(fitcurve) * abs(freq[0] - freq[1])
        spec_ax.plot(freqscale*freq[ar[0]:ar[1]], fitcurve, '--',
                     c='tab:red', alpha=0.7,
                     label='fit curve\nanalytical integral = %.4g uV^2 \n'
                           'sqrt root of integral = %g uV'%(popt[2],np.sqrt(popt[2])))
        #print("fit result = ", popt)
        print("Lorentzian integral = %.4g uV^2" % (popt[2]))
        print("sqrt root of Lorentzian integral = %.4g uV\n\n\n\n\n" % (np.sqrt(popt[2])))
    except:
        RuntimeError
    spec_ax.set_xlabel('absolute frequency / kHz')
    spec_ax.set_ylabel('Amplitude / $\mu V^2/Hz$')
    spec_ax.set_title('2021-11-30 exp num '+str(folde_arr[i]))
    spec_ax.legend(loc='upper right', fontsize=12)
    spec_ax.set_xlim(freqscale*freq[len(freq)//4], freqscale*freq[-1])
    spec_ax.set_ylim(-0.25e-6,4.5e-6)
    plt.grid()
    plt.savefig('D:\\Mainz\\CASPEr\\20211130 Keadata\\20211130\\' + str(folde_arr[i]) + '.png')
    plt.show()


    #plt.legend()
    del data, freq, spectrum, fig, gs, spec_ax, peakpara, popt, pcov































