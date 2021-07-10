import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

baselinepath = fpath+"Scan - lab course Tuesday, June 29, 2021 3_35 PM W. Europe Daylight Time\\"\
               +"100% or 0 Absorbance Baseline.Korrektur.Raw.csv"
baseline0629=np.loadtxt(baselinepath,delimiter=',',unpack=True,skiprows=2)

samplepath1= fpath+"Scan - lab course Wednesday, June 30, 2021 4_17 PM W. Europe Daylight Time\\"
sampstart1 = 563
sampend1 = 570
dataarr1 = []
for i in range(sampend1-sampstart1+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    print(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv")
    dataarr1.append(data)

baseline0630 = np.loadtxt(samplepath1+"100% or 0 Absorbance Baseline.Korrektur.Raw.csv",delimiter=',',unpack=True,skiprows=2)
dataarr1 = np.array(dataarr1)
print(dataarr1.shape)
#for i in range(4):
#    dataarr1[i+5][1, 0:] = -np.log(dataarr1[i+5][1, 0:]/100)

labelarr1 = np.array(["0 min",  # 563 0
                      "Low-V UV 5 min",  # 564
                      "UV 5 min",  # 565 2
                      "UV 10 min",  # 566
                      "UV 15 min",  # 567 4
                      "UV 15 min + High-V (2.8 V) Blue 5 min",  # 568 5
                      "Normal-V Blue 5 min",  # 569 6
                      "Blue 10 min",  # 570 7
                      "UV 15 min",  #  8
                      "UV 20 min",  #  9
                      "UV 20 min + High-V Blue 2 min",  #  10
                      "UV 20 min + High-V Blue 5 min",  #  11
                      "UV 20 min + Blue 5 min + Darkness 5 min",  #
                      "UV 20 min + Blue 5 min + Room light 1 min",  #
                      "UV 20 min + Blue 5 min + Room light 5 min", #
                      "UV 20 min + Blue 5 min + Room light 10 min"  #
                        ])

# sample02mM0630 blue 2.80 V


plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [2,3,4,5]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(190,610)
ax0.set_ylim(-0.025,0.445)
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Blue28V.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.295,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [2,3,4,5]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(412,454.4)
ax0.set_ylim(0.00220,0.00443)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Blue28Vzoomin2.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [2,3,4,5]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(305,326.6)
ax0.set_ylim(0.2438,0.2708)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Blue28Vzoomin1.png",format='png')
plt.show()

# sample02mM0630UVLowV
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [5,6,7]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(190,610)
ax0.set_ylim(-0.025,0.445)
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Blue.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.295,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [5,6,7]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(417,452.4)
ax0.set_ylim(0.00275,0.00468)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Bluezoomin2.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [5,6,7]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(305,326.6)
ax0.set_ylim(0.2387,0.2661)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Bluezoomin1.png",format='png')
plt.show()



# sample02mM0630UVLowV
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,1]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(190,610)
ax0.set_ylim(-0.025,0.425)
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630UVLowV.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.295,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,1]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(420,455.4)
ax0.set_ylim(0.00182,0.00380)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630UVLowVzoomin2.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,1]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0630[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(300,334.6)
ax0.set_ylim(0.2387,0.2837)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630UVLowVzoomin1.png",format='png')
plt.show()