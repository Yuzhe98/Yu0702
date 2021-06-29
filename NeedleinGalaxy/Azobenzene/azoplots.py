import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Saturday, April 3, 2021 5_10 PM W. Europe Daylight Time\\"
#sample0mM = samplepath1+"0.Probe.Raw"+".csv"
sample01mM = samplepath1+"0.1.Probe.Raw"+".csv"
#sample02mM = samplepath1+"0.2.Probe.Raw"+".csv"
sample03mM = samplepath1+"0.3.Probe.Raw"+".csv"
sample04mM = samplepath1+"0.4.Probe.Raw"+".csv"
sample05mM = samplepath1+"0.5.Probe.Raw"+".csv"

#data0mM = np.loadtxt(sample0mM,delimiter=',',unpack=True,skiprows=1)
data01mM = np.loadtxt(sample01mM,delimiter=',',unpack=True,skiprows=1)
#data02mM = np.loadtxt(sample02mM,delimiter=',',unpack=True,skiprows=1)
data03mM = np.loadtxt(sample03mM,delimiter=',',unpack=True,skiprows=1)
data04mM = np.loadtxt(sample04mM,delimiter=',',unpack=True,skiprows=1)
data05mM = np.loadtxt(sample05mM,delimiter=',',unpack=True,skiprows=1)
#print(data03mM.shape)

samplepath3 = fpath + "Scan - lab course Tuesday, March 30, 2021 10_22 PM W. Europe Daylight Time\\"
sample1mM = samplepath3+"Sample4.Probe.Raw"+".csv"
sample2mM = samplepath3+"Sample3.Probe.Raw"+".csv"
sample5mM = samplepath3+"Sample2.Probe.Raw"+".csv"
sample5mM2 = samplepath3+"Sample1.Probe.Raw"+".csv"
data1mM = np.loadtxt(sample1mM,delimiter=',',unpack=True,skiprows=1)
data2mM = np.loadtxt(sample2mM,delimiter=',',unpack=True,skiprows=1)
data5mM = np.loadtxt(sample5mM,delimiter=',',unpack=True,skiprows=1)
data5mM2 = np.loadtxt(sample5mM2,delimiter=',',unpack=True,skiprows=1)


samplepath4 = fpath + "Scan - lab course Sunday, June 27, 2021 3_05 PM W. Europe Daylight Time\\"
sample0mM = samplepath4+"Sample461.Probe.Raw"+".csv"
sample02mM = samplepath4+"Sample462.Probe.Raw"+".csv"
data0mM = np.loadtxt(sample0mM,delimiter=',',unpack=True,skiprows=2)
data02mM = np.loadtxt(sample02mM,delimiter=',',unpack=True,skiprows=2)


#dataarr = np.array([data0mM,data01mM,data02mM,data03mM,data04mM,data05mM,data1mM,data2mM,data5mM,data5mM2])
#labelarr = np.array(["0mM","data01mM","data02mM","data03mM","data04mM","data05mM","data1mM","data2mM","data5mM","data5mM2"])

dataarr = np.array([data0mM,data02mM,data1mM,data2mM,data5mM])
labelarr = np.array(["0 mM","0.2 mM","1 mM","2 mM","5 mM"])

fig = plt.figure(figsize=(3.8, 2.8))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)):
    ax0.plot(dataarr[i][0,:],100-100*np.exp(-dataarr[i][1,:]),label = labelarr[i])  #,c='grey'
#
ax0.set_xlabel('Wavelength / nm')  #, fontsize=14
ax0.set_ylabel('1-Transmittance / %')  #, fontsize=14Absorption
ax0.legend(loc='upper right')  # , fontsize=14
plt.show()

fig = plt.figure(figsize=(3.8, 2.8))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.133, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)):
    ax0.plot(dataarr[i][0,:],dataarr[i][1,:]/np.log(10),label = labelarr[i])  #,c='grey'
#
ax0.set_xlabel('Wavelength / nm')  #, fontsize=14
ax0.set_ylabel('Absorption')  #, fontsize=14
ax0.legend(loc='upper right')  # , fontsize=14
plt.show()