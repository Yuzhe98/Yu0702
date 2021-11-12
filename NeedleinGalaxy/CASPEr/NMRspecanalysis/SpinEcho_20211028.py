import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.gridspec as gridspec
def ExpDecay(t, A, T, off):
    return A * np.exp(-t / T) + off

filepath="D:\\Mainz\\CASPEr\\20211026 SpinEcho\\20211028 SpinEcho\\"
folder_arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
echotime_arr = np.array([30,40,50,60,80,100,120,150,180,240,300,320])  # in ms

dwellTime = 50.0/10**6  # in second
integral_arr = []
for i in range(len(folder_arr)):
    data = np.loadtxt(filepath+str(folder_arr[i])+"\\data.csv",delimiter=",")
    print(data.shape)
    data = np.transpose(data)
    print(data.shape)
    FIDcomplex = data[1]+1j*data[2]
    '''
    fig = plt.figure()  # figsize=(10, 10)
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    signal_ax = fig.add_subplot(gs[0, 0])
    signal_ax.plot(data[0],data[1],label="Real part")
    signal_ax.plot(data[0], data[2], label="Imaginary part",alpha=0.5)
    plt.show()
    '''
    FIDintegral = np.sum(np.abs(FIDcomplex)**2)*dwellTime
    integral_arr.append(FIDintegral)
    del data, FIDcomplex, FIDintegral

#print("data = {{%g, %g}, {%g, %g}, {%g, %g}, {800, %g}, {1000, %g}};" % (avg[0], avg[1], avg[2], avg[3], avg[4]))
'''
print("data = {")
for i in range(len(folder_arr)-1):
    print("{%g, Log[%g]}, " % (echotime_arr[i],integral_arr[i]))
print("{%g, Log[%g]}}; " % (echotime_arr[len(folder_arr)-1],integral_arr[len(folder_arr)-1]))
'''

p0 = (0.004, 100, 0) # start with values near those we expect
popt, pcov = scipy.optimize.curve_fit(ExpDecay, echotime_arr, integral_arr, p0)

time_arr = np.linspace(start=30, stop=320, num=1000,endpoint=True)

fig = plt.figure()  # figsize=(10, 10)
gs = gridspec.GridSpec(nrows=1, ncols=1)  #
fit_ax = fig.add_subplot(gs[0, 0])
fit_ax.scatter(echotime_arr,integral_arr,marker='x',label="data point")
fit_ax.plot(time_arr, ExpDecay(time_arr,popt[0],popt[1],popt[2]), '--', label="fit curve",c='gray')
fit_ax.set_xlabel("Echo time / ms")
fit_ax.set_ylabel("FID^2 integral")
fit_ax.set_title("Oct 28 SpinEcho experiment\nFit Result: %3f * Exp (-t/%3f) + %3f"%(popt[0],popt[1],popt[2]))
#fit_ax.axhline(y = integral_arr[0], color = 'green', linestyle = '-')
#fit_ax.text(110, integral_arr[0]+0.00005,"Amplitude = 0.004",color='green')
#fit_ax.text(110, integral_arr[0]/np.exp(1)+0.00005,"Amplitude = 0.004 / e",color='green')
#fit_ax.axhline(y = integral_arr[0]/np.exp(1), color = 'green', linestyle = '-')
plt.show()

print(popt)