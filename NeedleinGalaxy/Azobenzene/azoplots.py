import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fpath = "D:\\Mainz\\JGU\\Specmeter\\spec data\\"

samplepath1= fpath+"Scan - lab course Saturday, April 3, 2021 5_10 PM W. Europe Daylight Time\\"
sample0mM = samplepath1+"0.Probe.Raw"+".csv"
sample01mM = samplepath1+"0.1.Probe.Raw"+".csv"
#sample02mM = samplepath1+"0.2.Probe.Raw"+".csv"
sample03mM = samplepath1+"0.3.Probe.Raw"+".csv"
sample04mM = samplepath1+"0.4.Probe.Raw"+".csv"
sample05mM = samplepath1+"0.5.Probe.Raw"+".csv"
baseline0403 = np.loadtxt(samplepath1+"100% or 0 Absorbance Baseline.Korrektur.Raw.csv",delimiter=',',unpack=True,skiprows=1)
data0mM = np.loadtxt(sample0mM,delimiter=',',unpack=True,skiprows=1)+baseline0403
data01mM = np.loadtxt(sample01mM,delimiter=',',unpack=True,skiprows=1)+baseline0403
#data02mM = np.loadtxt(sample02mM,delimiter=',',unpack=True,skiprows=1)+baseline0403
data03mM = np.loadtxt(sample03mM,delimiter=',',unpack=True,skiprows=1)+baseline0403
data04mM = np.loadtxt(sample04mM,delimiter=',',unpack=True,skiprows=1)+baseline0403
data05mM = np.loadtxt(sample05mM,delimiter=',',unpack=True,skiprows=1)+baseline0403

samplepath3 = fpath + "Scan - lab course Tuesday, March 30, 2021 10_22 PM W. Europe Daylight Time\\"
sample1mM = samplepath3+"Sample4.Probe.Raw"+".csv"
sample2mM = samplepath3+"Sample3.Probe.Raw"+".csv"
sample5mM = samplepath3+"Sample2.Probe.Raw"+".csv"
sample5mM2 = samplepath3+"Sample1.Probe.Raw"+".csv"
baseline0330 = np.loadtxt(samplepath3+"100% or 0 Absorbance Baseline.Korrektur.Raw.csv",delimiter=',',unpack=True,skiprows=1)

data1mM = np.loadtxt(sample1mM,delimiter=',',unpack=True,skiprows=1)+ baseline0330
data2mM = np.loadtxt(sample2mM,delimiter=',',unpack=True,skiprows=1)+ baseline0330
data5mM = np.loadtxt(sample5mM,delimiter=',',unpack=True,skiprows=1)+ baseline0330
data5mM2 = np.loadtxt(sample5mM2,delimiter=',',unpack=True,skiprows=1)+ baseline0330


samplepath1= "D:\\Mainz\\JGU\\Specmeter\\spec data\\"+"Scan - lab course Wednesday, June 30, 2021 4_17 PM W. Europe Daylight Time\\"

baseline0630 = np.loadtxt(samplepath1+"100% or 0 Absorbance Baseline.Korrektur.Raw.csv",delimiter=',',unpack=True,skiprows=2)
data02mM = np.loadtxt(samplepath1+"Sample"+str(563)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)+baseline0630

#dataarr = np.array([data0mM,data01mM,data02mM,data03mM,data04mM,data05mM,data1mM,data2mM,data5mM,data5mM2])
#labelarr = np.array(["0mM","data01mM","data02mM","data03mM","data04mM","data05mM","data1mM","data2mM","data5mM","data5mM2"])

dataarr = np.array([data0mM,data01mM,data02mM,data1mM,data2mM,data5mM])
labelarr = np.array(["0 mM","0.1 mM","0.2 mM","1 mM","2 mM","5 mM"])
colors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#9D6F65']
#148 103 189
'''
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
'''
for i in range(len(dataarr)):
    print(dataarr[i][0,142])
    print(dataarr[i][0,80])
concentrations = np.array([0,0.1,0.2,1,2,5])
concentrations1 = np.array([0,0.1,0.2,1,2,5])
concentrations2 = np.array([0,0.1,0.2,1,2,5])
absp316nm = np.zeros(len(dataarr))
absp440nm = np.zeros(len(dataarr))

for i in range(len(dataarr)):
    absp316nm[i]=dataarr[i][1,142]
for i in range(len(dataarr)):
    absp440nm[i]=dataarr[i][1,80]

print("data = {{%g, %g}, {%g, %g}, {%g, %g}, {%g, %g}, {%g, %g}};"
      % (concentrations1[0],absp316nm[0],concentrations1[1],absp316nm[1],
         concentrations1[2],absp316nm[2],concentrations1[3],absp316nm[3],
         concentrations1[4],absp316nm[4]))
print("data = {{%g, %g}, {%g, %g}, {%g, %g}, {%g, %g}, {%g, %g}, {%g, %g}};"
      % (concentrations2[0],absp440nm[0],concentrations2[1],absp440nm[1],
         concentrations2[2],absp440nm[2],concentrations2[3],absp440nm[3],
         concentrations2[4],absp440nm[4],concentrations2[5],absp440nm[5]))

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(3.8, 2.8))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.193, top=0.90, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)-1):
    ax0.scatter(concentrations[i], absp316nm[i], marker="^",c=colors[i])  # ,c='grey'
#ax0.scatter(concentrations[:-1],absp316nm,marker="^")  #,c='grey'
ax0.plot(concentrations[:-1],0.045412+ 0.871632*concentrations[:-1],c='tab:gray')
#ax0.text(0,1.5,"0.045412 + 0.871632 $x$")
ax0.set_xlabel('Concentration / mM')  #, fontsize=14
ax0.set_ylabel('Absorption')  #, fontsize=14Absorption
ax0.set_title("316 nm",fontsize = 12)
plt.show()

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(3.8, 2.8))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.252, top=0.90, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)):
    ax0.scatter(concentrations[i], absp440nm[i], marker="D",c=colors[i])  # ,c='grey'
#ax0.scatter(concentrations[:-1],absp316nm,marker="^")  #,c='grey'
ax0.plot(concentrations[:],0.0232469+ 0.0215611*concentrations[:],c='tab:gray')
#ax0.text(-0.1,0.115,"0.0232469 + 0.0215611 $x$")
ax0.set_xlabel('Concentration / mM')  #, fontsize=14
ax0.set_ylabel('Absorption')  #, fontsize=14Absorption
ax0.set_title("440 nm",fontsize = 12)
plt.show()


plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)):
    ax0.plot(dataarr[i][0,:]/2,dataarr[i][1,:]/np.log(10),label = labelarr[i])  #,c='grey'
for i in range(len(dataarr)-1):
    ax0.scatter(dataarr[i][0, 142] / 2, dataarr[i][1, 142] / np.log(10), marker="^")  # ,c='grey'

ax0.set_xlabel('Wavelength / nm')  #, fontsize=14
ax0.set_ylabel('Absorption')  #, fontsize=14
ax0.legend(loc='upper right')  # , fontsize=14
plt.show()

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(3.8, 2.75))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.193,bottom=0.136, right=0.943, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in range(len(dataarr)):
    ax0.plot(dataarr[i][0,:]/2,dataarr[i][1,:]/np.log(10),label = labelarr[i])  #,c='grey'
for i in range(len(dataarr)):
    ax0.scatter(dataarr[i][0,80]/2,dataarr[i][1,80]/np.log(10),marker="D")  #,c='black'
ax0.set_xlim(368.8,525.7)
ax0.set_ylim(0.0125,0.1409)
#plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0630Blue28Vzoomin2.png",format='png')
plt.show()

