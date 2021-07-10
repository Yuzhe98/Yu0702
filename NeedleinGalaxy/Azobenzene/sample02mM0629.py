import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
def varmap(
        kappa316nmtrans = 0.871632,
        kappa316nmcis = np.ones(150),
        kappa440nmtrans = 0.0215611,
        kappa440nmcis = np.ones(100),
        kk2kk = np.ones(15)
):
    map = np.zeros((len(kappa316nmcis),len(kappa440nmcis)))
    selfvarmap = np.zeros((len(kappa316nmcis),len(kappa440nmcis)))
    for i in range(len(kappa316nmcis)):
        for j in range(len(kappa440nmcis)):
            kappatlide=(kappa316nmtrans - kappa316nmcis[i])/(kappa440nmtrans-kappa440nmcis[j])
            map[i][j]=np.sum((kappatlide-kk2kk)**2)
            selfvarmap[i][j]=np.var(kappatlide-kk2kk)
    return map, selfvarmap

def convarmap(
        kappa316nmtrans = 0.871632,
        kappa316nmcis = np.ones(150),
        kappa440nmtrans = 0.0215611,
        kappa440nmcis = np.ones(100),
        ab316 = np.ones(15),
        ab440=np.ones(15)
):
    map = np.zeros((len(kappa316nmcis),len(kappa440nmcis)))
    #selfvarmap = np.zeros((len(kappa316nmcis),len(kappa440nmcis)))
    for i in range(len(kappa316nmcis)):
        for j in range(len(kappa440nmcis)):
            kappatlide316=(kappa316nmtrans - kappa316nmcis[i])
            kappatlide440=(kappa440nmtrans - kappa440nmcis[j])
            deltacon316 = ab316/kappatlide316
            deltacon440= ab440/kappatlide440
            map[i][j]=np.sum(abs(deltacon316-deltacon440))
            #selfvarmap[i][j]=np.var(kappatlide-kk2kk)
    return map
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
baselinepath = fpath+"Scan - lab course Tuesday, June 29, 2021 3_35 PM W. Europe Daylight Time\\"\
               +"100% or 0 Absorbance Baseline.Korrektur.Raw.csv"
baseline0629=np.loadtxt(baselinepath,delimiter=',',unpack=True,skiprows=2)
'''
samplepath0630= fpath+"Scan - lab course Wednesday, June 30, 2021 4_17 PM W. Europe Daylight Time\\"
sampstart0630 = 563
sampend0630 = 570
dataarr0630 = []
for i in range(sampend0630-sampstart0630+1):
    data = np.loadtxt(samplepath0630+"Sample"+str(i+sampstart0630)+".Probe.Raw"+".csv",delimiter=',',unpack=True,skiprows=2)
    print(samplepath0630+"Sample"+str(i+sampstart0630)+".Probe.Raw"+".csv")
    dataarr0630.append(data)
dataarr0630= np.array(dataarr0630)
'''

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
                      "Darkness 2 min",  # 1
                      "UV 2 min + 1 min room light",
                      "UV 2 min + 2 min room light",
                      "UV 5 min",  # 4
                      "UV 5 min + 1 min weak room light",  # 5
                      "UV 10 min",  # 497 6
                      "UV 15 min",  # 498 7
                      "UV 15 min",  # 499 8
                      "UV 20 min",  # 500 9
                      "UV 20 min + High-V (3.30 V) Blue 2 min",  # 501 10
                      "UV 20 min + High-V (3.30 V) Blue 5 min",#"0 min",  # 502 11
                      "Darkness 5 min",  # 503 12
                      "Darkness 5 min + Room light 1 min",  # 504 13
                      "Darkness 5 min + Room light 5 min", # 505 14
                      "Darkness 5 min + Room light 10 min",  # 506 15
                    "Darkness 5 min + Room light 10 min \n+ Darkness 3 hour and 20 min"])  # 16

kappa316 = 0.871632*np.log(10)
off316 = 0.045412*np.log(10)
kappa440 = 0.0215611*np.log(10)
off440 = 0.0232469*np.log(10)
conc = 0.2
Ab316 = np.zeros(len(labelarr1))
Ab440 = np.zeros(len(labelarr1))
#FQ316 = np.zeros(len(labelarr1))
#FQ440 = np.zeros(len(labelarr1))
for i in range(len(labelarr1)):
    Ab316[i] = dataarr1[i][1,142]
    Ab440[i] = dataarr1[i][1, 80]
print("Ab316",Ab316)
print("max of Ab316 %g"%Ab316.max())
print("min of Ab316 %g"%Ab316.min())

print("Ab440",Ab440)
print("max of Ab440 %g"%Ab440.max())
print("min of Ab440 %g"%Ab440.min())

kmk2kmk0629 = (Ab316[2:]-Ab316[0])/(Ab440[2:]-Ab440[0])

#Ab316 = np.zeros(len(dataarr0630))
#Ab440 = np.zeros(len(dataarr0630))
#for i in range(len(dataarr0630)):
    #Ab316[i] = dataarr0630[i][1,142]
    #Ab440[i] = dataarr0630[i][1, 80]
#kmk2kmk0630 = (Ab316[1:]-Ab316[0])/(Ab440[1:]-Ab440[0])
kmk2kmk=np.zeros(len(kmk2kmk0629))
kmk2kmk[0:len(kmk2kmk0629)] =kmk2kmk0629
#kmk2kmk[len(kmk2kmk0629):]  =kmk2kmk0630
print("mkm = " ,kmk2kmk)
print("mean of kmk2kmk = %g"%np.mean(kmk2kmk))
print("std of kmk2kmk = %g"%np.std(kmk2kmk))

kappa316cis_arr = np.linspace(start=0.1*kappa316/9,stop=8*kappa316/9,num=150)
kappa440cis_arr = np.linspace(start=1.1*kappa440,stop=4*kappa440,num=200)
cisvar = np.zeros((len(kappa316cis_arr),len(kappa440cis_arr)))

xaxis = np.linspace(start=-1,stop=10,num=1000,endpoint=False)
colors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#9D6F65']
vmmap,smap=varmap(
        kappa316nmtrans = 0.871632,
        kappa316nmcis = kappa316cis_arr,
        kappa440nmtrans = 0.0215611,
        kappa440nmcis = kappa440cis_arr,
        kk2kk = kmk2kmk
)
concemap=convarmap(
        kappa316nmtrans=kappa316,
        kappa316nmcis=kappa316cis_arr,
        kappa440nmtrans=kappa440,
        kappa440nmcis=kappa440cis_arr,
        ab316 = Ab316[2:]-Ab316[0],
        ab440=Ab440[2:]-Ab440[0]
)
plt.rcParams.update({'font.size': 12})
fig = plt.figure()  #figsize=(7, 5.09)
gs = gridspec.GridSpec(nrows=1, ncols=2)  # , width_ratios=widths, height_ratios=heights
#fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
#for i in [5,9]:  #range(len(kkappas))
    #ax0.plot(xaxis, kkappas[i]*(xaxis-kappa440)+kappa316,label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
#for i in range(1,len(labelarr1)):
    #for j in range(i+1,len(labelarr1)):
        #x= (barr[i]-barr[j])/(karr[j]-karr[i])
        #ax0.scatter(x,karr[i]*x+barr[i])
        #print(x,karr[i]*x+barr[i])
#ax0.add_patch(Rectangle((1, 1), 2, 6,alpha=0.1))
#for i in range(len(kappa440arr1)):
#    ax0.hlines(i+1,kappa440arr2[i],kappa440arr1[i],color=colors[-1])
c = ax0.pcolormesh(kappa316cis_arr, kappa440cis_arr, vmmap.transpose(),
                   norm=LogNorm(vmin=vmmap.min(),vmax=vmmap.max()),cmap='plasma')
ax0.set_title('not whitened')
fig.colorbar(c, ax=ax0)
ax0.set_xlabel('$\kappa^{316nm}_{cis}$')  #, fontsize=18
ax0.set_ylabel('$\kappa^{440nm}_{cis}$')  #, fontsize=18
#ax0.set_xlim(-0.05,.250)
#ax0.legend(loc='upper right', fontsize=14)#
#plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\kappa-cis.png",format='png')
ax1 = fig.add_subplot(gs[0, 1])
c1 = ax1.pcolormesh(kappa316cis_arr, kappa440cis_arr, concemap.transpose(),norm=LogNorm(vmin=concemap.min(),vmax=concemap.max()),cmap='plasma')
fig.colorbar(c1, ax=ax1)
ax1.set_xlabel('$\kappa^{316nm}_{cis}$')  #, fontsize=18
ax1.set_ylabel('$\kappa^{440nm}_{cis}$')  #, fontsize=18
ax0.set_title('not whitened')
plt.show()




print("wl=")
print(dataarr1[0][0,142])
kapparr = np.zeros(len(labelarr1)-2)
print("data ={")
for i in range(len(labelarr1)-2):
    kapparr[i] = (dataarr1[i+2][1,142]-dataarr1[0][1,142])/(dataarr1[i+2][1,80]-dataarr1[0][1,80])
    print("{%g, %g}," % (i+1, kapparr[i]))
print("};")
print("std of kappas = %g"%np.std(kapparr))
print("mean of kappas = %g"%np.mean(kapparr))

deltac316 = np.zeros(len(labelarr1))
for i in range(len(labelarr1)):
    deltac316[i]=(dataarr1[i][1,142]-dataarr1[0][1,142])
print("max 316 nm = %g"%(np.max(deltac316)/(0.87162*(1/9-1))))
print("delta 316  = ",deltac316)

deltac440 = np.zeros(len(labelarr1))
for i in range(len(labelarr1)):
    deltac440[i]=(dataarr1[i][1,80]-dataarr1[0][1,80])
print("max 440 nm = %g"%(np.max(deltac440)/(0.0393832)))
print("max 440 nm = %g"%(np.max(deltac440)/(0.0477594)))
print("max 440 nm = %g"%(np.max(deltac440)/(0.0335066)))
print("delta 440  = ",deltac440)

print("ratio = ")
for i in range(len(labelarr1)-2):
    #deltac440[i]=(dataarr1[i][1,80]-dataarr1[0][1,80])
    print(deltac316[i+2]/deltac440[i+2])

for i in [0,11]:
    print(labelarr1[i])
    print(dataarr1[i][0,142])
    print((dataarr1[i][1, 142]))

    print(dataarr1[i][0,80])
    print((dataarr1[i][1, 80]))

print("baseline")
print(baseline0629[0,142])
print((baseline0629[1, 142]))
print(baseline0629[0,80])
print((baseline0629[1, 80]))

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13.82, 5.09))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.076, top=0.95, right=0.967,bottom=0.152, wspace=0.198, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
for i in [11,12,14,15,16]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
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
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00229,0.00501)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Roomzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [11,12,14,15,16]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
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
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
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
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00128,0.00436)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629Blue33Vzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.22,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,9,10,11]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
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
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
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
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(414,450)
ax0.set_ylim(0.00072,0.0041)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629UVzoomin2.png",format='png')
plt.show()

fig = plt.figure(figsize=(3, 2.3))
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
fig.subplots_adjust(left=0.23,bottom=0.157, right=0.971, top=0.974, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0])
for i in [0,4,6,7,9]:
    ax0.plot(dataarr1[i][0,:], (dataarr1[i][1,:]+baseline0629[1,:])/np.log(10),label = labelarr1[i])  #,label = labelarr[i]  ,c='grey'
ax0.set_xlim(307.6,324.89)
ax0.set_ylim(0.2237,0.2676)
plt.savefig("D:\\Mainz\\JGU\\Specmeter\\report\\sample02mM0629UVzoomin1.png",format='png')
plt.show()

