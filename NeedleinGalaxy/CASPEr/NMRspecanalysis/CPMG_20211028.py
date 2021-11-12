import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.gridspec as gridspec
def ExpDecay(t, A, T, off):
    return A * np.exp(-t / T) + off
def Exp2Decay(t, A1, T1, A2, T2, off):
    return A1 * np.exp(-t / T1) + A2 * np.exp(-t / T2) + off
filepath="D:\\Mainz\\CASPEr\\20211026 SpinEcho\\20211028 CPMG\\"
folder_arr = np.array([1,2,3,4,5,6,7,8])

temp_arr = [-57,-43.5,-37.9,-35.4 ,-32.2, -29.8 ,-28.7 ,-26.5,-26.1]
T2_arr = [20.66, 26.65 ,30.23 ,31.04 ,34.28, 36.79, 37.45 , 38.70, 40.27]
fig = plt.figure()  #figsize=(12, 4)
gs = gridspec.GridSpec(nrows=1, ncols=1)  #
temp_ax = fig.add_subplot(gs[0, 0])
temp_ax.scatter(temp_arr, T2_arr, marker='x', label="data point")
temp_ax.plot(temp_arr, T2_arr, )
temp_ax.set_xlabel("Temperature / C")
temp_ax.set_ylabel("T2 / ms")
#temp_ax.set_xlim(-60,0)
#temp_ax.set_ylim(0,90)
temp_ax.yaxis.set_label_position("right")
temp_ax.yaxis.tick_right()
temp_ax.set_title("")
temp_ax.grid()
plt.show()
for i in folder_arr:
    data = np.loadtxt(filepath + str(i) + "\\data.csv", delimiter=",")
    print(data.shape)
    data = np.transpose(data)
    print(data.shape)
    data[0] = data[0] / 1000

    p0 = (2, 30, 0)  # start with values near those we expect
    popt, pcov = scipy.optimize.curve_fit(ExpDecay, data[0], data[1], p0)
    p0_2 = (1, 60, 0.76, 12, 0)  # start with values near those we expect
    popt_2, pcov_2 = scipy.optimize.curve_fit(Exp2Decay, data[0], data[1], p0_2)

    time_arr = np.linspace(start=1.2, stop=264, num=1000, endpoint=True)

    fig = plt.figure()  #figsize=(12, 4)
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    fit_ax = fig.add_subplot(gs[0, 0])
    fit_ax.scatter(data[0], data[1], marker='x', label="data point")
    fit_ax.plot(time_arr, ExpDecay(time_arr, popt[0], popt[1], popt[2]), '--', label="fit curve", c='gray')
    fit_ax.set_xlabel("Time / ms")
    fit_ax.set_ylabel("Average echo amplitude")
    fit_ax.set_title("Fit Result: %.2f * Exp (-t/%.2f) + %.2f" % (popt[0], popt[1], popt[2]))
    '''
    fit_2_ax = fig.add_subplot(gs[0, 1])
    fit_2_ax.scatter(data[0], data[1], marker='x', label="data point")
    fit_2_ax.plot(time_arr, Exp2Decay(time_arr, popt_2[0], popt_2[1], popt_2[2], popt_2[3], popt_2[4]), '--',
                  label="fit curve", c='gray')
    fit_2_ax.set_xlabel("Time / ms")
    fit_2_ax.set_ylabel("Average echo amplitude")
    fit_2_ax.set_title("Fit Result: %.2f * Exp (-t/%.2f) + %.2f * Exp (-t/%.2f) + %.2f" % (
    popt_2[0], popt_2[1], popt_2[2], popt_2[3], popt_2[4]))
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_arr, np.exp(-abs(time_arr-time_arr[len(time_arr)//2])/20)*np.sin(0.3*time_arr))
    '''
    plt.show()

    print(popt)
