import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, report_fit, Model
from scipy.optimize import curve_fit
import h5py
import sys
import scipy as sp


def loadStream(filename):
    """load file
    Args:
        fileName (string): file name with path

    Returns:
        array: dataX, dataY, pulseData, frequency, timeconstant, filterorder
    """    
    with h5py.File(filename, 'r') as dataFile:
        dataX = dataFile['000/dev4434/demods/0/sample/x'][:]
        dataY = dataFile['000/dev4434/demods/0/sample/y'][:]
        pulseData = dataFile["000/dev4434/demods/0/sample/auxin0"][:]
        #frequency = dataFile["000/dev4434/demods/0/sample/frequency"][0]
        #timeconstant = dataFile["000/dev4434/demods/0/timeconstant/value"][0]
        #filterorder = dataFile["000/dev4434/demods/0/order/value"][0]
    return dataX, dataY, pulseData#, frequency, timeconstant, filterorder


def findpulse(dataX = [], dataY = [], pulseData = [], trigger = 0):
    """
    Args:
        dataX (np.list, optional): . Defaults to [].
        dataY (np.list, optional): . Defaults to [].
        pulseData (np.list, optional): TTL signal. Defaults to [].
        trigger (int, optional): [0: start of TTL trigger, 1: end of TTL, 2: start of pulse, 3: end of pulse]. Defaults to 0.
    Returns:
        [array]: [all indices indicate the end the pulse]
    """
    startofpulse = []
    endofpulse = []
    if trigger == 0:
        trigger_val = 2
        startofpulse = np.flatnonzero((pulseData[1:] > trigger_val) & (pulseData[:-1] < trigger_val))
        print('startofpulse='+str(startofpulse))
        endofpulse = startofpulse + pointsofpulselength
        print('endofpulse='+str(endofpulse))
    elif trigger == 1:
        trigger_val = 2.5
        endofpulse = np.flatnonzero((pulseData[1:] < trigger_val) & (pulseData[:-1] > trigger_val))
        print('endofpulse='+str(endofpulse))
    elif trigger == 2:
        dataXY = np.abs(dataX + 1j*dataY)
        trigger_val = 0.04
        startofpulse = np.flatnonzero((dataXY[1:] > trigger_val) & (dataXY[:-1] < trigger_val))+1
        endofpulse = startofpulse + pointsofpulselength
    elif trigger == 3:
        dataXY = np.abs(dataX + 1j*dataY)
        trigger_val = 0.04
        endofpulse = np.flatnonzero((dataXY[1:] < trigger_val) & (dataXY[:-1] > trigger_val))+1
    elif trigger == 4:
        trigger_val = np.amax(pulseData)/2.
        startofpulse = np.flatnonzero((pulseData[1:] > trigger_val) & (pulseData[:-1] < trigger_val))
        endofpulse = np.flatnonzero((pulseData[1:] < trigger_val) & (pulseData[:-1] > trigger_val))
    else:
        raise ValueError('trigger value out of range')
    return startofpulse, endofpulse


def Lorentzian(x, center, gamma, A, offset):
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))


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

    frequencies = np.fft.fftfreq(len(data_x), d=1 / samprate)  # Set d to dwell time in s
    #print(frequencies.shape)
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter=='on':
        filtercomp /=LIAFilterPSD(
            frequency = frequencies,
            taun = DTRCfilter_Tc,
            order = DTRCfilter_order)

    FFT = np.fft.fft((data_x + 1j * data_y) * windowfunction)
    PSD = 10.0**(attenuation/10.) * 2.0 * np.abs(FFT) ** 2 / (S2 * samprate) * filtercomp

    frequencies += dfreq
    return np.sort(frequencies), PSD[np.argsort(frequencies)]


def signalamp(
        mu0=1.25663706212e-6,  # in {\henry.\metre^{-1}
        mup=1.41060679736e-26,
        temp=-50,
        tempunit='C',
        sampvol = 1.4e-3,  # in 'L'
        sampdens=26.5,  # in mol/L
        NA=6.023e23,
        Bfield = 13e-4,  # in T
        SQUID_num='C6L1W',  # 'C73L1'
        L=953e-9,  # in H
):
    return 0

def pulseNMRplot(
        filename,
        dfreq = 0,  # in Hz
        samprate = 13e3,  # in Hz
        pulselength = 3.0,
        acqdelay = 0.1,
        acqtime = 1,
        showtimedomain = True,
        showacqdata = True,
        showfreqdomain = True,
        spectype ='Flux',
        Mf=31706,  # feedback sensitivity which can be found in the SQUID specifications. For the SQUID we usually use, M_f = 31 706 \phi_0 / A
        Rf=10e3,  # in Ohm
        frequnit = 'kHz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
        ampunit = 'muPhi',
        singlePSD_arr = [0],
        stddev_range = [35e3,37e3],
        left_spc=0.1,
        top_spc=1-0.1,
        right_spc=1-.05,
        bottom_spc=.1,
        xgrid_spc=.3,
        ygrid_spc=.2,
        verbose=True
):
    '''
    filename,
    dfreq = 0,  # in Hz
        samprate = 13e3,  # in Hz
        pulselength = 3.0,
        acqdelay = 0.1,
        acqtime = 1,
        showtimedomain = True,
        showacqdata = True,
        showfreqdomain = True,
    spectype:
        'PSD', 'ASD', 'Flux'
    Mf:
        feedback sensitivity which can be found in the SQUID specifications. For the SQUID we usually use, M_f = 31 706 \phi_0 / A
    Rf=10e3,  # in Ohm
    frequnit = 'kHz':
        The unit of frequency. 'Hz', 'kHz', 'MHz', 'GHz' and 'THz' are available.
    ampunit:
        unit of the amplitude of the spectrum. It can be 'V', 'muV', 'Phi', or 'muPhi'. The default unit is 'muPhi'

    singlePSD_arr = [0, 1, 2],
    stddev_range:
        Compute the standard deviation within the given frequency range. The unit of it is Hz.
    left_spc=0.1,
    top_spc=1-0.1,
    right_spc=1-.05,
    bottom_spc=.1,
    xgrid_spc=.3,
    ygrid_spc=.2,
    :return:
    '''

    dataX, dataY, pulseData = loadStream(filename)  # , frequency, timeconstant, filterorder
    if verbose:
        print('Time constant: ')
    dataXY = np.abs(dataX + 1j * dataY)
    startofpulse, endofpulse = findpulse(dataX, dataY, pulseData, 4)

    if len(startofpulse) == 0:
        raise ValueError('len(startofpulse) == 0')
    if len(endofpulse) == 0:
        raise ValueError('len(endofpulse) == 0')

    if startofpulse[0] < endofpulse[0]:
        startofpulse = startofpulse[1:]
    if endofpulse[-1] > startofpulse[-1]:
        endofpulse = endofpulse[:-1]
    #print(startofpulse.shape)
    #print(endofpulse.shape)
    if len(startofpulse) == 0:
        raise ValueError('len(startofpulse) == 0 after correction')
    if len(endofpulse) == 0:
        raise ValueError('len(endofpulse) == 0 after correction')
    pointsofpulse = endofpulse[1]-endofpulse[0]
    signaltime_arr = np.linspace(start=0, stop=len(pulseData) / samprate, num=len(pulseData), endpoint=False,
                                 dtype=float)
    numofpulse = len(endofpulse)
    #pointsofpulselength = int(pulselength * samprate)
    pointsofacqdelay = int(acqdelay * samprate)
    pointsofacqtime = int(acqtime * samprate)
    if (pointsofacqdelay+pointsofacqtime) > np.min(startofpulse-endofpulse):
        raise ValueError('(pointsofacqdelay+pointsofacqtime) > min(endofpulse-startofpulse)')
    startofacq = endofpulse + pointsofacqdelay
    acq_arr = np.array([startofacq,startofacq+pointsofacqtime]).transpose()

    plt.style.use('seaborn-colorblind')
    # plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    # 'tab:blue' 'tab:orange''tab:green' 'tab:red''tab:purple''tab:brown''tab:pink''tab:gray''tab:olive''tab:cyan'

    for i in singlePSD_arr:
        frequencies, PSD = stdLIAPSD(
            data_x=dataX[acq_arr[i, 0]:acq_arr[i, 1] + 1],
            data_y=dataY[acq_arr[i, 0]:acq_arr[i, 1] + 1],
            dataunit='V',  # 'muV', 'nV'
            samprate=samprate,  # in Hz
            dfreq=dfreq,  # in Hz
            attenuation=6,  # in dB. Power ratio (10^(attenuation/10))
            window='rectangle',  # Hanning, Hamming, Blackman
            DTRCfilter='on',
            DTRCfilter_Tc=2.548e-5,
            DTRCfilter_order=8)

        fig = plt.figure(figsize=(18, 9))  #
        gs = gridspec.GridSpec(nrows=3, ncols=2)  #
        fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
                            bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        displaytime = [max(0,acq_arr[i, 0]-2*pointsofpulse), min(acq_arr[i, 1]+2*pointsofpulse, len(signaltime_arr))]
        print(displaytime)
        pulse_ax = fig.add_subplot(gs[0, 0])
        pulse_ax.plot(signaltime_arr[displaytime[0]:displaytime[1]], pulseData[displaytime[0]:displaytime[1]],
                      label="pulse signal", c='tab:purple')
        pulse_ax.set_ylabel('Voltage / V')
        pulse_ax.set_xlim(signaltime_arr[displaytime[0]], signaltime_arr[displaytime[1]])

        dataX_ax = fig.add_subplot(gs[1, 0])
        dataX_ax.plot(signaltime_arr[displaytime[0]:displaytime[1]], dataX[displaytime[0]:displaytime[1]],
                      label="LIA X data",
                      c='tab:green')
        dataX_ax.plot(signaltime_arr[acq_arr[i, 0]:acq_arr[i, 1]],dataX[acq_arr[i, 0]:acq_arr[i, 1]],
                      label="LIA X data for PSD",
                      c='tab:cyan', alpha=0.5)

        dataX_ax.set_ylabel('Voltage / V')
        dataX_ax.set_xlim(signaltime_arr[displaytime[0]], signaltime_arr[displaytime[1]])

        dataY_ax = fig.add_subplot(gs[2, 0])
        dataY_ax.plot(signaltime_arr[displaytime[0]:displaytime[1]], dataY[displaytime[0]:displaytime[1]], label="LIA Y data",
                      c='tab:brown')
        dataY_ax.plot(signaltime_arr[acq_arr[i, 0]:acq_arr[i, 1]], dataY[acq_arr[i, 0]:acq_arr[i, 1]],
                      label="LIA Y data for PSD",
                      c='tab:cyan', alpha=0.5)
        dataY_ax.set_ylabel('Voltage / V')
        dataY_ax.set_xlim(signaltime_arr[displaytime[0]], signaltime_arr[displaytime[1]])
        dataY_ax.set_xlabel('time / s')

        pulse_ax.legend(loc='upper right')
        dataX_ax.legend(loc='upper right')
        dataY_ax.legend(loc='upper right')

        #frequencies+=dfreq
        # Compute the standard deviation
        stdstart = np.argmin(abs(frequencies-stddev_range[0]))
        stdend = np.argmin(abs(frequencies - stddev_range[1]))
        if abs(stdstart-stdend)<=10:
            print("abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation")
        stddev = np.std(PSD[stdstart:stdend])

        if frequnit == 'kHz':
            freqfactor = 1e-3
        elif frequnit == 'MHz':
            sfreqfactor = 1e-6
        elif frequnit == 'GHz':
            freqfactor = 1e-9
        elif frequnit == 'THz':
            freqfactor = 1e-12
        elif frequnit != 'Hz':
            freqfactor = 1
        else:
            raise ValueError('frequnit wrong')

        if ampunit == 'muV' or ampunit == 'microV' or ampunit == 'uV' or ampunit == 'muv' or ampunit == 'microv' or ampunit == 'uv':
            ampfactor = 1e6
            densityunit = '$\mu V/\sqrt{Hz}$'
        elif ampunit == 'muPhi' or ampunit == 'microPhi' or ampunit == 'uPhi' or ampunit == 'muphi' or ampunit == 'microphi' or ampunit == 'uphi':
            ampfactor = 1e6
            densityunit = '$\mu \Phi_{0}/\sqrt{Hz}$'
        elif ampunit == 'V' or ampunit == 'v':
            ampfactor = 1
            densityunit = '$V/\sqrt{Hz}$'
        elif ampunit == 'Phi' or ampunit == 'phi':
            ampfactor = 1
            densityunit = '$\Phi_{0}/\sqrt{Hz}$'
        else:
            raise ValueError('ampunit wrong')

        if spectype == 'PSD':
            PSD_ax = fig.add_subplot(gs[:, 1])
            PSD_ax.plot(freqfactor*frequencies, PSD, label="Power Spectrum Density", c='tab:blue')
            PSD_ax.plot(freqfactor * frequencies[stdstart:stdend], PSD[stdstart:stdend],
                        label='Standard deviation = %.2f'%(stddev)+'$V^{2}/Hz$'+'\n(Computed based on PSD data)', c='tab:red')
            PSD_ax.set_xlabel('absolute frequency / ' + frequnit)
            PSD_ax.set_ylabel('PSD / $V^{2}/Hz$')
            PSD_ax.set_yscale("log")
            PSD_ax.legend(loc='upper right')
        elif spectype == 'ASD':
            ASD_ax = fig.add_subplot(gs[:, 1])
            ASD_ax.plot(freqfactor*frequencies, ampfactor*np.sqrt(PSD), label="Amplitude Spectrum Density", c='tab:blue')
            ASD_ax.plot(freqfactor * frequencies[stdstart:stdend], ampfactor*np.sqrt(PSD[stdstart:stdend]),
                  label='Standard deviation = %.2f '% (ampfactor*np.sqrt(stddev))+ densityunit +'\n(Computed based on PSD data)' , c='tab:red')
            ASD_ax.set_xlabel('absolute frequency / ' + frequnit)
            ASD_ax.set_ylabel('ASD / ' + densityunit)
            ASD_ax.set_yscale("log")
            ASD_ax.legend(loc='upper right')
        elif spectype == 'Flux':
            FSD_ax = fig.add_subplot(gs[:, 1])
            FSD_ax.plot(freqfactor*frequencies, ampfactor*Mf/Rf*np.sqrt(PSD), label="Flux Spectrum Density", c='tab:blue')
            FSD_ax.plot(freqfactor * frequencies[stdstart:stdend],
                        ampfactor*Mf/Rf* np.sqrt(PSD[stdstart:stdend]),
                        label='Standard deviation = %.2f ' % (ampfactor*Mf/Rf*np.sqrt(stddev))+densityunit + '\n(Computed based on PSD data)',
                        c='tab:red')
            FSD_ax.set_xlabel('absolute frequency / ' + frequnit)
            FSD_ax.set_ylabel('FSD / ' + densityunit)
            FSD_ax.set_yscale("log")
            FSD_ax.legend(loc='upper right')
        else:
            raise ValueError('spectype set wrong')

        # plt.legend('upper right')  # 'upper left', 'upper right', 'lower left', 'lower right'
        fig.suptitle('Single shot\n' + filename)
        #plt.tight_layout()
        plt.grid()
        plt.show()
        del frequencies, PSD, fig, gs, stdstart, stdend, stddev


    PSD = np.zeros(pointsofacqtime)
    #print(PSD.shape)
    for i in range(numofpulse):
        #print(singleshot)
        frequencies, singlePSD = stdLIAPSD(
            data_x = dataX[acq_arr[i,0]:acq_arr[i,1]],
            data_y = dataY[acq_arr[i,0]:acq_arr[i,1]],
            dataunit = 'V',  # 'muV', 'nV'
            samprate = samprate,  # in Hz
            dfreq = dfreq,  # in Hz
            attenuation = 6,  # in dB. Power ratio (10^(attenuation/10))
            window = 'rectangle',  # Hanning, Hamming, Blackman
            DTRCfilter='on',
            DTRCfilter_Tc=2.5486726372037083E-5,
            DTRCfilter_order=8
        )
        #print(singlePSD.shape)
        PSD += singlePSD/numofpulse
        del singlePSD


    fig = plt.figure(figsize=(18, 9))  #
    gs = gridspec.GridSpec(nrows=3, ncols=2)  #
    fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
                        bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    if showtimedomain:
        pulse_ax = fig.add_subplot(gs[0, 0])
        pulse_ax.plot(signaltime_arr, pulseData, label="pulse signal", c='tab:purple')
        pulse_ax.set_ylabel('Voltage / V')
        pulse_ax.set_xlim(signaltime_arr[0],signaltime_arr[-1])

        dataX_ax = fig.add_subplot(gs[1, 0])
        dataX_ax.plot(signaltime_arr, dataX, label="LIA X", c='tab:green')
        dataX_ax.set_ylabel('Voltage / V')
        dataX_ax.set_xlim(signaltime_arr[0], signaltime_arr[-1])

        dataY_ax = fig.add_subplot(gs[2, 0])
        dataY_ax.plot(signaltime_arr, dataY, label="LIA Y", c='tab:brown')
        dataY_ax.set_ylabel('Voltage / V')
        dataY_ax.set_xlim(signaltime_arr[0], signaltime_arr[-1])

        dataY_ax.set_xlabel('time / s')
        pulse_ax.legend(loc='upper right')
        dataX_ax.legend(loc='upper right')
        dataY_ax.legend(loc='upper right')

        if showacqdata:
            acqsignaltime = []
            acqdataX = []
            acqdataY = []
            for i in range(numofpulse):
                acqsignaltime.append(signaltime_arr[acq_arr[i, 0]:acq_arr[i, 1] + 1])
                acqdataX.append(dataX[acq_arr[i, 0]:acq_arr[i, 1] + 1])
                acqdataY.append(dataY[acq_arr[i, 0]:acq_arr[i, 1] + 1])
            dataX_ax.plot(acqsignaltime, acqdataX, label="LIA X data for PSD", c='tab:cyan', alpha=0.5)
            dataY_ax.plot(acqsignaltime, acqdataY, label="LIA Y data for PSD", c='tab:cyan', alpha=0.5)

        #frequencies+=dfreq
        stdstart = np.argmin(abs(frequencies - stddev_range[0]))
        stdend = np.argmin(abs(frequencies - stddev_range[1]))
        if abs(stdstart - stdend) <= 10:
            print("abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation")
        stddev = np.std(PSD[stdstart:stdend])

        if frequnit == 'kHz':
            freqfactor = 1e-3
        elif frequnit == 'MHz':
            freqfactor = 1e-6
        elif frequnit == 'GHz':
            freqfactor = 1e-9
        elif frequnit == 'THz':
            freqfactor = 1e-12
        elif frequnit != 'Hz':
            freqfactor = 1
        else:
            raise ValueError('frequnit wrong')

        if ampunit == 'muV' or ampunit == 'microV' or ampunit == 'uV' or ampunit == 'muv' or ampunit == 'microv' or ampunit == 'uv':
            ampfactor = 1e6
            densityunit = '$\mu V/\sqrt{Hz}$'
        elif ampunit == 'muPhi' or ampunit == 'microPhi' or ampunit == 'uPhi' or ampunit == 'muphi' or ampunit == 'microphi' or ampunit == 'uphi':
            ampfactor = 1e6
            densityunit = '$\mu \Phi_{0}/\sqrt{Hz}$'
        elif ampunit == 'V' or ampunit == 'v':
            ampfactor = 1
            densityunit = '$V/\sqrt{Hz}$'
        elif ampunit == 'Phi' or ampunit == 'phi':
            ampfactor = 1
            densityunit = '$\Phi_{0}/\sqrt{Hz}$'
        else:
            raise ValueError('ampunit wrong')

    if showfreqdomain:
        if spectype == 'PSD':
            PSD_ax = fig.add_subplot(gs[:, 1])
            PSD_ax.plot(freqfactor * frequencies, PSD, label="Power Spectrum Density", c='tab:blue')
            PSD_ax.plot(freqfactor * frequencies[stdstart:stdend], PSD[stdstart:stdend],
                        label='Standard deviation = %.2f' % (stddev) + '$V^{2}/Hz$' + '\n(Computed based on PSD data)',
                        c='tab:red')
            PSD_ax.set_xlabel('absolute frequency / ' + frequnit)
            PSD_ax.set_ylabel('PSD / $V^{2}/Hz$')
            PSD_ax.set_yscale("log")
            PSD_ax.legend(loc='upper right')
        elif spectype == 'ASD':
            ASD_ax = fig.add_subplot(gs[:, 1])
            ASD_ax.plot(freqfactor * frequencies, ampfactor * np.sqrt(PSD), label="Amplitude Spectrum Density",
                        c='tab:blue')
            ASD_ax.plot(freqfactor * frequencies[stdstart:stdend], ampfactor * np.sqrt(PSD[stdstart:stdend]),
                        label='Standard deviation = %.2f ' % (
                                    ampfactor * np.sqrt(stddev)) + densityunit + '\n(Computed based on PSD data)',
                        c='tab:red')
            ASD_ax.set_xlabel('absolute frequency / ' + frequnit)
            ASD_ax.set_ylabel('ASD / ' + densityunit)
            ASD_ax.set_yscale("log")
            ASD_ax.legend(loc='upper right')
        elif spectype == 'Flux':
            FSD_ax = fig.add_subplot(gs[:, 1])
            FSD_ax.plot(freqfactor * frequencies, ampfactor * Mf / Rf * np.sqrt(PSD), label="Flux Spectrum Density",
                        c='tab:blue')
            FSD_ax.plot(freqfactor * frequencies[stdstart:stdend],
                        ampfactor * Mf / Rf * np.sqrt(PSD[stdstart:stdend]),
                        label='Standard deviation = %.2f ' % (ampfactor * Mf / Rf * np.sqrt(
                            stddev)) + densityunit + '\n(Computed based on PSD data)',
                        c='tab:red')
            FSD_ax.set_xlabel('absolute frequency / ' + frequnit)
            FSD_ax.set_ylabel('FSD / ' + densityunit)
            FSD_ax.set_yscale("log")
            FSD_ax.legend(loc='upper right')
        else:
            raise ValueError('spectype set wrong')
    #plt.legend('upper right')  # 'upper left', 'upper right', 'lower left', 'lower right'
    fig.suptitle('All shots\n'+filename)
    plt.grid()
    plt.show()

    del fig, gs


pulseNMRplot(
        filename="D:\\Mainz\\CASPEr\\20211112 Lowfield NMR\\data\\17\\stream_000/stream_00000.h5",
        dfreq = 15e3,  # in Hz
        samprate = 13.39e3,  # in Hz
        pulselength = 1,
        acqdelay = 0.1,
        acqtime = 1,
        showtimedomain = True,
        showacqdata = True,
        showfreqdomain = True,
        spectype='Flux',
        Mf=31706,  # feedback sensitivity
        Rf=10e3,  # in Ohm
        frequnit='kHz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
        ampunit='muPhi',  # 'V', 'muV', 'muPhi'
        singlePSD_arr=[2],
        stddev_range=[15e3, 17e3],
)