import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Tuesday, June 29, 2021 3_35 PM W. Europe Daylight Time\\"

sampstart1 = 491
sampend1 = 507
dataarr1 = []
for i in range(sampend1-sampstart1+1):
    data = np.loadtxt(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    print(samplepath1+"Sample"+str(i+sampstart1)+".Probe.Raw"+".csv")
    dataarr1.append(data)

dataarr1 = np.array(dataarr1)
print(dataarr1.shape)
#for i in range(4):
#    dataarr1[i+5][1, 0:] = -np.log(dataarr1[i+5][1, 0:]/100)

labelarr1 = np.array(["UV 0 min",  # 0
                      "Darkness 2 min",  #
                      "UV 2 min + 1 min room light",
                      "UV 2 min + 2 min room light",
                      "UV 5 min",  # 4
                      "UV 5 min + 1 min weak room light",  # 5
                      "UV 10 min",  # 497 6
                      "UV 15 min",  # 498 7
                      "UV 15 min",  # 499 8
                      "UV 20 min",  # 500 9
                      "UV 20 min + High-V (3.30 V) Blue 2 min",  # 501 10
                      "0 min",#"UV 20 min + High-V (3.30 V) Blue 5 min",  # 502 11
                      "Darkness 5 min",  # 503
                      "Room light 1 min",  # 504
                      "Room light 5 min", # 505
                      "Room light 10 min"  # 506
                        ])
# room
labelarr1 = np.array(["UV 0 min",  # 0
                      "Darkness 2 min",  #
                      "UV 2 min + 1 min room light",
                      "UV 2 min + 2 min room light",
                      "UV 5 min",  # 4
                      "UV 5 min + 1 min weak room light",  # 5
                      "UV 10 min",  # 497 6
                      "UV 15 min",  # 498 7
                      "UV 15 min",  # 499 8
                      "UV 20 min",  # 500 9
                      "UV 20 min + High-V (3.30 V) Blue 2 min",  # 501 10
                      "0 min",#"UV 20 min + High-V (3.30 V) Blue 5 min",  # 502 11
                      "Darkness 5 min",  # 503
                      "Darkness 5 min + Room light 1 min",  # 504
                      "Darkness 5 min + Room light 5 min", # 505
                      "Darkness 5 min + Room light 10 min",  # 506
                    "Darkness 5 min + Room light 10 min \n+ Darkness 3 hour and 20 min"])
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [11,12,14,15,16]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Room.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [11,12,14,15,16]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00229,0.00501)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Roomzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [11,12,14,15,16]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(296,334.89)
ax0.set_ylim(0.1919,0.2639)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Roomzoomin1.png",format='png')
plt.show()

# 3.30 V blue
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,9,10,11]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Blue33V.png",format='png')
plt.show()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,9,10,11]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00128,0.00436)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Blue33Vzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,9,10,11]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(307.6,324.89)
ax0.set_ylim(0.2237,0.2676)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Blue33Vzoomin1.png",format='png')
plt.show()







# UV
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,4,6,7,9]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlabel('Wavelength / nm')  #, fontsize=18
ax0.set_ylabel('Absorption')  #, fontsize=18
ax0.legend(loc='upper right', fontsize=14)#
plt.show()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.23,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,4,6,7,9]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00072,0.0041)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629UVzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.23,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,4,6,7,9]:
    ax0.plot(dataarr1[i][0,:], dataarr1[i][1,:]/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(307.6,324.89)
ax0.set_ylim(0.2237,0.2676)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629UVzoomin1.png",format='png')
plt.show()

