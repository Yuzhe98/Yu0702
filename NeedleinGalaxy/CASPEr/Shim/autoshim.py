import pyautogui
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, Model
import h5py
from scipy.optimize import curve_fit


class ShimAppControl():
    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    F = 0
    G = 0
    H = 0
    def __init__(self, a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E = e
        self.F = f
        self.G = g
        self.H = h
    __dic = { "A": (610, 158),
                "B": (610, 208),
                "C": (610, 268),
                "D": (610, 328),
                "E": (610, 380),
                "F": (610, 435),
                "G": (610, 490),
                "H": (610, 545),
                "ok": (540, 714),
                }
    def startapp(self):
        pyautogui.moveTo(614, 1179, duration = 1)
        pyautogui.click(614, 1179) # start cryogenic app
        time.sleep(10)
        pyautogui.click(585, 625) # click "shims"
        time.sleep(10)
    def switchtocontroller(self):
        pyautogui.click(614, 1179)
        pyautogui.moveTo(681, 1042, duration = 1)
        pyautogui.click(681, 1042)
    def applycurrent(self):
        self.switchtocontroller()
        for i in vars(ShimAppControl):
            if(len(i) == 1):
                pyautogui.click( self.__dic[i])
                pyautogui.press('backspace', presses=10) #clear all data
                pyautogui.press('del', presses=10)
                pyautogui.typewrite(str(self.__dict__[i]))
                time.sleep(0.1)
        pyautogui.click(self.__dic["ok"])
        time.sleep(1)

class NMRAppControl():
    def start1pulse(self):
        pyautogui.moveTo(423, 29)
        # pyautogui.click(423, 29)
        pyautogui.click(423, 29) # click KEA-NMR
        time.sleep(0.5)
        pyautogui.click(420, 100) # click "1pulse"
    def run(self):
        pyautogui.moveTo(554, 1189)
        pyautogui.click(554, 1189) # click "KEA-NMR app"
        pyautogui.click(195, 950)  # click "run"
def Lorentzian(x, center, gamma, A, offset):
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))

# shim log file
sourceFile = open('shimming_record20210824.csv', 'a+')
sys.stdout = sourceFile

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

shimcurrent=ShimAppControl()
# shimcurrent.startapp()

initialexpnumber = 253
itertable = [0, 1, -1, 2, -2]
expnumber = initialexpnumber
for a in itertable:
    for d in itertable:
        for b in itertable:
            for c in itertable:     
                shimcurrent=ShimAppControl(a, b, c, d)
                shimcurrent.applycurrent()
                time.sleep(2)
                nmrobject = NMRAppControl()
                nmrobject.run()
                time.sleep(12)
                sys.stdout.write('%d, %5.3f, %5.3f, %5.3f, %5.3f, ' % (expnumber, a, b, c, d))
                fit(expnumber)
                sys.stdout.write("\n")
                expnumber = expnumber + 1



# shimcurrent.applycurrent()

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 16
