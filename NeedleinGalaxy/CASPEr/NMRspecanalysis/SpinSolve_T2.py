import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.gridspec as gridspec
def ExpDecay(t, A, T, off):
    return A * np.exp(-t / T) + off

print(np.linspace(start=0, stop=5000, num=10,endpoint=True))
echotime_arr = np.linspace(start=0.2, stop=5000, num=10,endpoint=True)  # in ms

integral_arr = np.array([1300,964,723,537,393,285,200,137,89,56])

p0 = (1300, 50, 0) # start with values near those we expect
popt, pcov = scipy.optimize.curve_fit(ExpDecay, echotime_arr, integral_arr, p0)

time_arr = np.linspace(start=0, stop=5000, num=1000,endpoint=True)

fig = plt.figure()  # figsize=(10, 10)
gs = gridspec.GridSpec(nrows=1, ncols=1)  #
fit_ax = fig.add_subplot(gs[0, 0])
fit_ax.scatter(echotime_arr,integral_arr,marker='x',label="data point",c="orange")
fit_ax.plot(time_arr, ExpDecay(time_arr,popt[0],popt[1],popt[2]), '--', label="fit curve",c='gray')
fit_ax.set_xlabel("Echo time / ms")
fit_ax.set_ylabel("Integral over signal peak")
fit_ax.set_title("SpinSolve SpinEcho experiment\nFit Result: %.2f * Exp (-t/%.2f) %.2f"%(popt[0],popt[1],popt[2]))
#fit_ax.axhline(y = integral_arr[0], color = 'green', linestyle = '-')
#fit_ax.text(110, integral_arr[0]+0.00005,"Amplitude = 1300",color='red')
#fit_ax.text(110, integral_arr[0]/np.exp(1)+0.00005,"Amplitude = 1300 / e",color='red')
#fit_ax.axhline(y = integral_arr[0]/np.exp(1), color = 'green', linestyle = '-')
plt.show()

print(popt)