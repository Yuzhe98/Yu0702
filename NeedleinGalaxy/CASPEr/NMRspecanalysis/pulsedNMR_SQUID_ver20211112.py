import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, report_fit, Model
from scipy.optimize import curve_fit
import h5py
import sys
import scipy as sp

acqtime = 1.0            # in s      #1000ms
samplefreq = 13.29e3    # in Hz     #13.39kHz
pulselength = 1         # in s      #10**6us
acqdelay = 0.1         # in s      #100000us
pointsofpulselength = int(pulselength * samplefreq)
pointsofacqdelay = acqdelay * samplefreq
pointsofacqtime = acqtime * samplefreq
filename = "D:\\Mainz\\CASPEr\\20211112 Lowfield NMR\\datastream_000/stream_00000.h5"

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


def findendofpulse(dataX = [], dataY = [], pulseData = [], trigger = 0):
    """
    Args:
        dataX (np.list, optional): . Defaults to [].
        dataY (np.list, optional): . Defaults to [].
        pulseData (np.list, optional): TTL signal. Defaults to [].
        trigger (int, optional): [0: start of TTL trigger, 1: end of TTL, 2: start of pulse, 3: end of pulse]. Defaults to 0.
    Returns:
        [array]: [all indices indicate the end the pulse]
    """    
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
    return endofpulse


def FT(dataX, dataY):
    """FT
    Args:
        dataX (np.array): 
        dataY (np.array): 
    Returns:
        (np.array, np.array): frequencies, spectrum (sorted)
    """
    
    # do FFT
    FIDComplex = dataX + 1j*dataY
    FFTOfBlock = np.fft.fft(FIDComplex)

    numberOfPoints = len(dataX)   
    PSD = 2*np.abs(FFTOfBlock)**2/(numberOfPoints*samplefreq)
    #PSD = (np.abs(FFTOfBlock)**2)*numberOfPoints/(samplefreq)
    frequencies = np.fft.fftfreq(numberOfPoints, d = 1/samplefreq) # Set d to dwell time in s

    return np.sort(frequencies), PSD[np.argsort(frequencies)]


def Lorentzian(x, center, gamma, A, offset):
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))


def LIAFilterPSD(frequency, Tn, order):
    return np.abs(1/(1 + 2*np.pi*1j*frequency*Tn)**order)**2


def plot(n, caption=''):
    global pointsofpulselength
    global filename
    sys.stdout = open(filename[:11]+'result.txt', 'w+')
    outputfile = filename[:-3]+".txt"

    plt.style.use('seaborn-colorblind')
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 16

    dataX, dataY, pulseData = loadStream(filename)
    dataXY = np.abs(dataX + 1j*dataY)
    endofpulse = findendofpulse(dataX, dataY, pulseData, 0)
    start = (int)(endofpulse[0] + pointsofacqdelay)
    stop = (int)(start + pointsofacqtime)
    print('#elements in EoP: '+str(len(endofpulse)))
    plotEndIndex = 100000
    plt.plot(pulseData[:plotEndIndex])
    #plt.plot(dataX)
    #plt.plot(dataY, alpha = 0.5)
    plt.plot(dataXY[:plotEndIndex])
    plt.plot(endofpulse[0], 1, "o")
    plt.plot(start, 1, "o")
    plt.plot(stop, 1, "o")
    #plt.xlim(0,50000)
    plt.show()
    #plt.close()
    
    df = pd.DataFrame({"dataX" : dataX, "dataY" : dataY, "pulseData" : pulseData})
    df.to_csv("FID.txt", index=False, header=False)
    
    PSD = np.zeros(int(pointsofacqtime))
    #print("debug mark")
    for i in endofpulse[:-1]: # if the last FID length goes beyond the data length, use -2 instead of -1
        start = (int)(i + pointsofacqdelay)
        #print(start)
        stop = (int)(start + pointsofacqtime)
        #print(stop)
        #print(dataX[start: stop])
        frequencies, PSD = FT(dataX[start: stop], dataY[start: stop])
        PSD = PSD + PSD / len(endofpulse)
        
    #print(PSD)
    # save spectra
    df = pd.DataFrame({"frequencies" : frequencies, "spectra" : PSD})
    df.to_csv(outputfile, index=False, header=False)
    
    # fit
    # popt, pcov = curve_fit(Lorentzian, frequencies, spectra, p0=(frequencies[np.argmax(spectra)], 1, 100000, 0))
    # plt.plot(frequencies, Lorentzian(frequencies, *popt), 'g--',)

    print('file = '+filename)
    print('acqtime = %.6fs' %acqtime)
    print('samplefreq = %.1fHz' %samplefreq)
    print('pulselength = %.6fs' %pulselength)
    print('acqdelay = %.6fs' %acqdelay)
    # print('fit: center=%5.3f, linewidth=%5.3f, area=%5.3f' % ( popt[0], popt[1], popt[2]))
    
    pos1 , t = min(enumerate(frequencies), key=lambda x: abs(-600 - x[1]))
    pos2 , t = min(enumerate(frequencies), key=lambda x: abs(x[1]))
    sum = 0
    print(frequencies[2]-frequencies[1])
    print(frequencies[pos1], frequencies[pos2])
    #for i in range(pos1, pos2):
    #    sum = sum + PSD[i] * (frequencies[2]-frequencies[1]) 
    plt.plot(frequencies, np.sqrt(PSD))
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude spectral density ($ \phi_0^2 / Hz$)")
    plt.ylabel("Amplitude spectral density ($V^2/Hz$)")
    plt.title(filename)
    plt.savefig(filename[:11]+"spectrum.pdf")
    plt.show()
    plt.close()
    return sum  # return area


# this is specifically for amplitude sweep
# 300us no.46 is too noisy?
# dict = {34: '25us', 38: '100us', 39: '125us', 40: '150us', 41: '175us', 3: '300us', 48: '350us', 49: '375us', 50: '400us', 51: '425us', 52: '450us', 54: '500us'}

# area, t, err = [], [], []

# error = plot(999)
# for i in dict.keys():
#     area.append(plot(i,dict[i]))
#     t.append(float(dict[i][:-2]))
#     err.append(error)
# plt.errorbar(t, area, yerr = err, fmt='o')
# plt.plot(t, area)
# plt.title('pulse duration sweep')
# plt.ylabel('area from -600Hz to 0Hz($\phi_0^2$)')
# plt.xlabel('pulse duration(us)')    
# plt.savefig('durationsweep.pdf')
# plt.legend()
# plt.savefig('durationsweepstacked.pdf')
# for i in range(3, 9):
#     plot(i)


plot(1)