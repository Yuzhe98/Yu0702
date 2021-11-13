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
        array: dataX, dataY, pulseData
    """    
    with h5py.File(filename, 'r') as dataFile:
        dataX = dataFile['000/dev4434/demods/0/sample/x'][:]
        dataY = dataFile['000/dev4434/demods/0/sample/y'][:]
        pulseData = dataFile["000/dev4434/demods/0/sample/auxin0"][:]
    return dataX, dataY, pulseData


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


def LIAFilterPSD(frequency, Tn, order):
    return np.abs(1/(1 + 2*np.pi*1j*frequency*Tn)**order)**2

def stdLIAPSD(
        data_x,
        data_y,
        dataunit = 'V',  # 'muV', 'nV'
        samprate = 1e3,  # in Hz
        dfreq = 0,  # in Hz
        attenuation = 6,  # in dB. Power ratio (10^(attenuation/10)). Positive value means signal was attenuated beforehand.
        window = 'rectangle'  # Hanning, Hamming, Blackman
):
    '''
    Based on https://holometer.fnal.gov/GH_FFT.pdf
    :param data_x:
    :param data_y:
    :param dataunit:
    :param samprate:
    :param dfreq:
    :param attenuation:
    :param window:
    :return:
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
    FFT = np.fft.fft((data_x + 1j * data_y) * windowfunction)
    PSD = 10.0**(attenuation/10.) * 2.0 * np.abs(FFT) ** 2 / (S2 * samprate)
    frequencies = np.fft.fftfreq(len(data_x), d=1 / samprate)  # Set d to dwell time in s
    frequencies += dfreq
    return np.sort(frequencies), PSD[np.argsort(frequencies)]

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
        PSDorASD = 'ASD',
        frequnit = 'Hz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
        singlePSD_arr = [0, 1, 2],
        left_spc=0.1,
        top_spc=1-0.1,
        right_spc=1-.05,
        bottom_spc=.1,
        xgrid_spc=.3,
        ygrid_spc=.2,
):
    '''

    :return:
    '''

    dataX, dataY, pulseData = loadStream(filename)
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
        singlefrequencies, singlePSD = stdLIAPSD(data_x=dataX[acq_arr[i, 0]:acq_arr[i, 1] + 1],
                                                 data_y=dataY[acq_arr[i, 0]:acq_arr[i, 1] + 1],
                                                 dataunit='V',  # 'muV', 'nV'
                                                 samprate=samprate,  # in Hz
                                                 dfreq=0,  # in Hz
                                                 attenuation=6,  # in dB. Power ratio (10^(attenuation/10))
                                                 window='rectangle'  # Hanning, Hamming, Blackman
                                                 )

        fig = plt.figure(figsize=(18, 9))  #
        gs = gridspec.GridSpec(nrows=3, ncols=2)  #
        fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
                            bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        displaytime = [max(0,acq_arr[i, 0]-pointsofpulse//2), min(acq_arr[i, 1]+pointsofpulse//2, len(signaltime_arr))]
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

        if frequnit == 'kHz':
            singlefrequencies /= 1e3
        elif frequnit == 'MHz':
            singlefrequencies /= 1e6
        elif frequnit == 'GHz':
            singlefrequencies /= 1e9
        elif frequnit == 'THz':
            singlefrequencies /= 1e12
        elif frequnit != 'Hz':
            raise ValueError('frequnit wrong')

        if PSDorASD == 'PSD':
            PSD_ax = fig.add_subplot(gs[:, 1])
            PSD_ax.plot(singlefrequencies, singlePSD, label="PSD", c='tab:blue')
            PSD_ax.set_xlabel('frequency / ' + frequnit)
            PSD_ax.set_ylabel('PSD / $V^2/Hz$')
            PSD_ax.set_yscale("log")
            PSD_ax.legend(loc='upper right')
        elif PSDorASD == 'ASD':
            ASD_ax = fig.add_subplot(gs[:, 1])
            ASD_ax.plot(singlefrequencies, np.sqrt(singlePSD), label="ASD", c='tab:blue')
            ASD_ax.set_xlabel('frequency / ' + frequnit)
            ASD_ax.set_ylabel('ASD / $V/\sqrt{Hz}$')
            ASD_ax.set_yscale("log")
            ASD_ax.legend(loc='upper right')
        else:
            raise ValueError('PSDorASD set wrong')

        # plt.legend('upper right')  # 'upper left', 'upper right', 'lower left', 'lower right'
        fig.suptitle('Single shot\n' + filename)
        plt.grid()
        plt.show()
        del singlefrequencies, singlePSD, fig, gs


    PSD = np.zeros(pointsofacqtime)
    #print(PSD.shape)
    for i in range(numofpulse):
        #print(singleshot)
        frequencies, singlePSD = stdLIAPSD(data_x = dataX[acq_arr[i,0]:acq_arr[i,1]],
        data_y = dataY[acq_arr[i,0]:acq_arr[i,1]],
        dataunit = 'V',  # 'muV', 'nV'
        samprate = samprate,  # in Hz
        dfreq = 0,  # in Hz
        attenuation = 6,  # in dB. Power ratio (10^(attenuation/10))
        window = 'rectangle'  # Hanning, Hamming, Blackman
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
        if frequnit=='kHz':
            frequencies /= 1e3
        elif frequnit=='MHz':
            frequencies /= 1e6
        elif frequnit=='GHz':
            frequencies /= 1e9
        elif frequnit=='THz':
            frequencies /= 1e12
        elif frequnit != 'Hz':
            raise ValueError('frequnit wrong')

    if showfreqdomain:
        if PSDorASD == 'PSD':
            PSD_ax = fig.add_subplot(gs[:, 1])
            PSD_ax.plot(frequencies, PSD, label="PSD", c='tab:blue')
            PSD_ax.set_xlabel('frequency / ' + frequnit)
            PSD_ax.set_ylabel('PSD / $V^2/Hz$')
            PSD_ax.set_yscale('log')
            PSD_ax.legend(loc='upper right')
        elif PSDorASD == 'ASD':
            ASD_ax = fig.add_subplot(gs[:, 1])
            ASD_ax.plot(frequencies, np.sqrt(PSD), label="ASD", c='tab:blue')
            ASD_ax.set_xlabel('frequency / ' + frequnit)
            ASD_ax.set_ylabel('ASD (Sqrt of averaged PSD) / $V/\sqrt{Hz}$')
            ASD_ax.set_yscale('log')
            ASD_ax.legend(loc='upper right')
        else:
            raise ValueError('PSDorASD set wrong')
    #plt.legend('upper right')  # 'upper left', 'upper right', 'lower left', 'lower right'
    fig.suptitle('All shots\n'+filename)
    plt.grid()
    plt.show()

    del fig, gs


pulseNMRplot(
        filename="D:\\Mainz\\CASPEr\\20211112 Lowfield NMR\\data\\stream_002/stream_00000.h5",
        dfreq = 30e3,  # in Hz
        samprate = 26.79e3,  # in Hz
        pulselength = 1,
        acqdelay = 0.1,
        acqtime = 1,
        showtimedomain = True,
        showacqdata = True,
        showfreqdomain = True,
        PSDorASD = 'ASD',
        frequnit = 'Hz',  # in Hz by default. 'kHz' 'MHz' 'GHz' 'THz'
        singlePSD_arr = [0, 1, 2]
)