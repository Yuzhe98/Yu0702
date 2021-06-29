import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Tuesday, June 22, 2021 6_36 PM W. Europe Daylight Time\\"

sampstart1 = 421
sampend1 = 428
dataarr1 = []
for i in range(sampend1-sampstart1+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    dataarr1.append(data)

dataarr1=  np.array(dataarr1)
print(dataarr1.shape)

#dataarr1 = np.array([data0mM,data01mM,data02mM,data03mM,data04mM,data05mM,data1mM,data2mM,data5mM,data5mM2])
#labelarr = np.array(["0mM","data01mM","data02mM","data03mM","data04mM","data05mM","data1mM","data2mM","data5mM","data5mM2"])

labelarr1 = np.array(["UV 0 min","1 mM","2 mM","5 mM"])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr1)):
    ax0.plot(dataarr1[i][0,:],100-100*np.exp(-dataarr1[i][1,:]),label = "UV " + str(10*i)+" min")  #,label = labelarr[i]  ,c='grey'
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.vlines(376,-10,45,colors='grey')
ax0.text(380,22," nm")
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('1-Transmittance / %', fontsize=12)  #
#ax0.set_xlim(-2,2)
ax0.set_ylim(-2,105)
#ax1.set_title('', fontsize=18)
ax0.legend(loc='upper right', fontsize=12)

plt.show()

print(dataarr1[0][0,112])
timeseries = 10*np.linspace(start=0,stop=len(dataarr1)-1,num=len(dataarr1),endpoint=True)
absoarr1 = np.zeros(len(dataarr1))
for i in range(len(dataarr1)):
    absoarr1[i] = 100-100*np.exp(-dataarr1[i][1,112])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(timeseries,absoarr1)
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.set_xlabel('UV exposure time / min', fontsize=12)  #
ax0.set_ylabel('1-Transmittance / %', fontsize=12)  #
#ax0.set_xlim(-2,2)
#ax0.set_ylim(-2,105)
#ax1.set_title('', fontsize=18)
#ax0.legend(loc='upper right', fontsize=12)

plt.show()


sampstart2 = 386
sampend2 = 391
dataarr2 = []
for i in range(sampend2-sampstart2+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart2)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    dataarr2.append(data)

dataarr2=  np.array(dataarr2)
print(dataarr2.shape)
labelarr = np.array(["0.2 mM","1 mM","2 mM","5 mM"])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr2)):
    ax0.plot(dataarr2[i][0,:],100-100*np.exp(-dataarr2[i][1,:]),label = str(i))  #  ,c='grey'
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('1-Transmittance / %', fontsize=12)  #
#ax0.set_xlim(-2,2)
#ax0.set_ylim(-.12,5)
#ax1.set_title('', fontsize=18)
ax0.legend(loc='upper right', fontsize=12)

#ax1 = fig.add_subplot(gs[0, 1])
#ax1.plot(freq1[0:len(freq1)//2]+0.832,spec1[0:len(freq1)//2],c='#1f77b4')  #/ (200.0 + 200.0 * i)
#ax1.plot(freq1[len(freq1)//2+1:]+0.832,spec1[len(freq1)//2+1:],c='#1f77b4')
#ax1.set_xlabel('Frequency [kHz]')  #, fontsize=16
#ax1.set_ylabel('Amplitude [$\mu$V/kHz]')  #, fontsize=16
#ax1.set_xlim(-4,4)
#ax1.set_ylim(-4,57.5)
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr2)):
    ax0.plot(dataarr2[i][0,:],100-100*np.exp(-dataarr2[i][1,:]),label = str(i))  #  ,c='grey'
for i in range(len(dataarr1)):
    ax0.plot(dataarr1[i][0,:],100-100*np.exp(-dataarr1[i][1,:]),label = str(i+100))  #  ,c='grey'
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('1-Transmittance / %', fontsize=12)  #
#ax0.set_xlim(-2,2)
#ax0.set_ylim(-.12,5)
#ax1.set_title('', fontsize=18)
ax0.legend(loc='upper right', fontsize=12)

#ax1 = fig.add_subplot(gs[0, 1])
#ax1.plot(freq1[0:len(freq1)//2]+0.832,spec1[0:len(freq1)//2],c='#1f77b4')  #/ (200.0 + 200.0 * i)
#ax1.plot(freq1[len(freq1)//2+1:]+0.832,spec1[len(freq1)//2+1:],c='#1f77b4')
#ax1.set_xlabel('Frequency [kHz]')  #, fontsize=16
#ax1.set_ylabel('Amplitude [$\mu$V/kHz]')  #, fontsize=16
#ax1.set_xlim(-4,4)
#ax1.set_ylim(-4,57.5)
plt.show()