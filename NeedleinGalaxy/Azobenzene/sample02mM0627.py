import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Sunday, June 27, 2021 3_05 PM W. Europe Daylight Time\\"

sampstart1 = 463
sampend1 = 471
dataarr1 = []
for i in range(sampend1-sampstart1+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    print(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv")
    dataarr1.append(data)

dataarr1 = np.array(dataarr1)
print(dataarr1.shape)
for i in range(4):
    dataarr1[i+5][1, 0:] = -np.log(dataarr1[i+5][1, 0:]/100)

labelarr1 = np.array(["UV 0 min","UV 5 min","UV 10 min","UV 15 min","0 min","UV 20 min + Blue 5 min",\
                      "UV 20 min + Blue 10 min","UV 20 min + Blue 10 min + UV 5 min","UV 20 min + Blue 10 min + UV 10 min"])

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [5,6,7,8]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [5,6,7,8]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#ax0.set_xlabel('Wavelength / nm')  #, fontsize=12
#ax0.set_ylabel('Absorption')  #, fontsize=12
#ax0.legend(loc='upper right', fontsize=12)
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr1)):
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#ax0.plot(data01mM)  # ,c='grey''#1f77b4'
#ax0.vlines(376,-10,45,colors='grey')
#ax0.text(380,22," nm")
ax0.set_xlabel('Wavelength / nm', fontsize=12)  #
ax0.set_ylabel('Absorption', fontsize=12)  #
#ax0.set_xlim(-2,2)
#ax0.set_ylim(-2,105)
ax0.legend(loc='upper right', fontsize=12)
plt.show()




'''
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
plt.show()
'''