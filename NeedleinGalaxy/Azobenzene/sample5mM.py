import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Monday, June 21, 2021 3_16 PM W. Europe Daylight Time\\"

sampstart1 = 376
sampend1 = 391#391
dataarr1 = []
for i in [374,375,377,378,379,380,381,383,384,387,388,389,390,391]:
    data = np.loadtxt(samplepath1+"Sample"+str(i)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    dataarr1.append(data)

dataarr1=  np.array(dataarr1)
print(dataarr1.shape)
#dataarr1 = np.array([data0mM,data01mM,data02mM,data03mM,data04mM,data05mM,data1mM,data2mM,data5mM,data5mM2])
#labelarr = np.array(["0mM","data01mM","data02mM","data03mM","data04mM","data05mM","data1mM","data2mM","data5mM","data5mM2"])
labelarr1 = np.array(["UV 0 min","UV 1 min","UV 11 min","UV 21 min","UV 31 min","UV 41 min","UV 51 min"\
                      ,"UV 51 min + Blue 10 min","UV 51 min + Blue 20 min","UV 51 min + Blue 30 min"\
                      ,"UV 51 min + Blue 40 min","UV 51 min + Blue 50 min","UV 51 min + Blue 60 min"\
                      ,"UV 51 min + Blue 70 min"])
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))  #
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,6,12]:  #
    ax0.plot(dataarr1[i][0,:],dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'#ax0.legend(loc='upper right')
ax0.set_xlabel('Wavelength / nm')  #
ax0.set_ylabel('Absorption')  #1-Transmittance / %
ax0.legend(loc='upper right', fontsize=14)
plt.show()

fig = plt.figure(figsize=(2.2, 2.2))  #
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,6,12]:  #
    ax0.plot(dataarr1[i][0,:],dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#ax0.legend(loc='upper right')
#ax0.set_xlabel('Wavelength / nm')  #
#ax0.set_ylabel('Absorption')  #1-Transmittance / %
ax0.set_xlim(360,420)
ax0.set_ylim(0.1,0.5)
#plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample5mMzoomin1.png",format='png')
plt.show()


fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(7):  # [4,5,6,8,10,12]
    ax0.plot(dataarr1[i][0,:],100-100*np.exp(-dataarr1[i][1,:]),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#dataarr1[i][1,:]/np.log(10)
ax0.legend(loc='upper right', fontsize=12)
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('1-T', fontsize=12)  #1-Transmittance / %
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(7):  # [4,5,6,8,10,12]
    ax0.plot(dataarr1[i][0,:],dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.legend(loc='upper right', fontsize=12)
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('Absorption', fontsize=12)  #1-Transmittance / %
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,6,12]:  #
    ax0.plot(dataarr1[i][0,:],100-100*np.exp(-dataarr1[i][1,:]),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.legend(loc='upper right', fontsize=12)
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('1-T', fontsize=12)  #1-Transmittance / %
#ax0.set_title('[0,6,7,8,9,10,11,12]', fontsize=12)
plt.show()






fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr1)):
    ax0.plot(dataarr1[i][0,:],100-100*np.exp(-dataarr1[i][1,:]),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#
#ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
#ax0.set_ylabel('Absorption', fontsize=12)  #1-Transmittance / %
ax0.legend(loc='upper right', fontsize=12)
plt.show()

print(dataarr1[0][0,83])
timeseries = 10*np.linspace(start=0,stop=len(dataarr1)-1,num=len(dataarr1),endpoint=True)
absoarr1 = np.zeros(len(dataarr1))
for i in range(len(dataarr1)):
    absoarr1[i] = dataarr1[i][1,83]  #100-100*np.exp(-dataarr1[i][1,112])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(timeseries,absoarr1)
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.set_xlabel('UV irradiation time / min', fontsize=12)  #
ax0.set_ylabel('', fontsize=12)  #1-Transmittance / %
plt.show()


sampstart2 = 381
sampend2 = 391
dataarr2 = []
for i in range(sampend2-sampstart2+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart2)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    dataarr2.append(data)

dataarr2=  np.array(dataarr2)
print(dataarr2.shape)
labelarr2 = np.array(["UV 0 min","UV 1 min","UV 11 min","UV 21 min","UV 31 min","UV 41 min","UV 51 min"])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr2)):
    ax0.plot(dataarr2[i][0,:],dataarr2[i][1,:],label = labelarr2[i])  # 100-100*np.exp(-dataarr2[i][1,:]) ,c='grey'
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('Absorption', fontsize=12)  #1-Transmittance
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