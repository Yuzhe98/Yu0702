import pyautogui
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, Model
import h5py
from scipy.optimize import curve_fit
def fit(expnumber):
    data = np.loadtxt('1pulse/'+str(expnumber)+'/data.csv', unpack=True, delimiter=',')
    FIDComplex = data[1] + 1j*data[2]

    spectrum = np.fft.fft(FIDComplex, norm='ortho')
    spectrum = np.abs(spectrum)**2

    frequencies = -np.fft.fftfreq(len(spectrum), d=0.05) # Set d to dwell time in ms
    centerEstimate = frequencies[np.argmax(spectrum)]
    
    fitModel = Model(Lorentzian)
    gamma=0.1
    # result = fitModel.fit(spectrum, x=frequencies, center= centerEstimate, gamma=0.1, A=0.1, offset = 0)
    popt, pcov = curve_fit(Lorentzian, frequencies, spectrum, p0=(frequencies[np.argmax(spectrum)], 0.1, 0.1, 0))
    plt.plot(frequencies, Lorentzian(frequencies, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
    plt.plot(frequencies, spectrum)
    # gamma equals FWHM for a Lorenztian
    plt.title('fit: center=%5.3f, linewidth=%5.3f, area=%5.3f, expnumber=%d' % ( popt[0], popt[1], popt[2], expnumber ) )
    plt.savefig('1pulse/'+str(expnumber)+'/fit.pdf')
    plt.close()
    plt.figure()
    # center, FWHM, area
    sys.stdout.write('%5.3f, %5.3f, %5.3f' % ( popt[0], popt[1], popt[2] ))
    return popt[1]