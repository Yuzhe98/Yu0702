import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, Model

#%matplotlib inline

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['font.size'] = 16

# fitModel = Model(lossFunction)
# result = fitModel.fit(yLoadedData, x=xLoadedData, gamma=0.1, N0=6, A=20)
# print(result.fit_report())
# result.plot()
def exponentialDecay(x, y0, tau, offset):
    return y0 * np.exp(-x / tau) + offset

def Gaussian(x, sigma, mu, A, offset):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def GaussianBeam(x, waist, mu, power, offset):
    return ((2*power) / (np.pi*waist**2))*np.exp(-2*(x-mu)**2/(waist**2))

def Gaussian2D(x, y, centerX, centerY, sigmaX, sigmaY, angle, amplitude, offset):
    return offset + amplitude*(np.exp(-0.5*((x-centerX)*np.cos(angle)-(y-centerY)*np.sin(angle))**2/(sigmaX**2)-0.5*((y-centerY)*np.cos(angle)+(x-centerX)*np.sin(angle))**2/(sigmaY**2)))

def lossFunction(x, gamma, N0, A):
    return N0 * np.exp(-gamma*x) / (1 + N0*A*(1-np.exp(-gamma*x))/gamma)

def Lorentzian(x, center, gamma, A, offset):
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))

fpath = "D:\\Mainz\\CASPEr\\13C\\20210608-140329-1D PROTON-sample_13C_enriched_methanol\\"
#data = np.loadtxt(fpath+'spectrum.1d')  # , unpack=True, delimiter=','

with open(fpath+'spectrum.1d') as f:
    content = f.readlines()
    print(content)

# Show the file contents line by line.
# We added the comma to print single newlines and not double newlines.
# This is because the lines contain the newline character '\n'.
for line in content:
    print(line)

FIDComplex = data[1] + 1j*data[2]
FIDComplex = FIDComplex[:1023]
dwellt= .2  # Set dwell time in ms
spectrum = np.fft.fft(FIDComplex)  # , norm='ortho'
spectrum = np.abs(spectrum)*dwellt  #**2
frequencies = -np.fft.fftfreq(len(spectrum), d=dwellt) # Set d to dwell time in ms

centerEstimate = frequencies[np.argmax(spectrum)]
fitModel = Model(Lorentzian)
result = fitModel.fit(spectrum, x=frequencies, center=-0.1, gamma=0.05, A=3, offset = 0)  # center= centerEstimate
print(result.fit_report())
result.plot(datafmt='-')
plt.xlabel('Frequency[kHz]')
plt.ylabel('Amp[$\mu$V/kHz]')
plt.show()
#plt.xlim(-0.6, 0.6)