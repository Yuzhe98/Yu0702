import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec

filepath="D:\\Mainz\\CASPEr\\20211026 SpinEcho\\20211028 CPMG\\9\\"
filename = "data.csv"
filename2 = "data2.csv"
data = np.loadtxt(filepath + filename, delimiter=",")
data2 = np.loadtxt(filepath + filename2, delimiter=",")
print(data.shape)
print(data2.shape)
print(data2[0].shape)

print("data[0] ", data[0])
print("Average of data2[0] = ", np.mean(data2[0]))
print("Average of square of data2[0] = ", np.mean(data2[0]**2))
print("SQ root of average of square of data2[0] = ", np.sqrt(np.mean(data2[0]**2)))
print("Average of abs of data2[0] = ", np.mean(abs(data2[0])))

for i in range(256):  # len(data[:,0])
    print(data[i],np.mean(data2[i]),data[i,1]/np.mean(data2[i]))
