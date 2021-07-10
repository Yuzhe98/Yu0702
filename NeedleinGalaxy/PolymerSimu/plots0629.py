import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def Rplot(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210629test\\R_statistic\\overnight\\",
        fname=np.array([#"Endtoend_radius#plen100#numstep100000#time",
                        "Endtoend_radius#plen200#numstep100000#time",
                        "Endtoend_radius#plen400#numstep100000#time",
                        "Endtoend_radius#plen600#numstep100000#time",
                        "Endtoend_radius#plen800#numstep100000#time",
                        "Endtoend_radius#plen1000#numstep100000#time"
                        ]),
        #"Endtoend_radius#plen10#numstep100000#time","Endtoend_radius#plen20#numstep100000#time",\
        #                "Endtoend_radius#plen50#numstep100000#time", "Endtoend_radius#plen100#numstep100000#time", \
        #                "Endtoend_radius#plen150#numstep100000#time",
        #"Endtoend_radius#plen250#numstep100000#time", "Endtoend_radius#plen300#numstep100000#time",
        #"Endtoend_radius#plen500#numstep100000#time",
        #"Endtoend_radius#plen700#numstep100000#time",
        #
        #

        fname1=np.array(["Endtoend_radius#plen200#numstep50000#time", "Endtoend_radius#plen400#numstep50000#time",
                        "Endtoend_radius#plen600#numstep50000#time", "Endtoend_radius#plen800#numstep50000#time",
                        "Endtoend_radius#plen1000#numstep50000#time"]),
        fname2=np.array(["Endtoend_radius#plen200#numstep100000#time", "Endtoend_radius#plen400#numstep100000#time",
                        "Endtoend_radius#plen600#numstep100000#time", "Endtoend_radius#plen800#numstep100000#time",
                        "Endtoend_radius#plen1000#numstep100000#time"]),
        fname3=np.array(["Endtoend_radius#plen200#numstep1000000#time", "Endtoend_radius#plen400#numstep1000000#time",
                        "Endtoend_radius#plen600#numstep1000000#time", "Endtoend_radius#plen800#numstep1000000#time",
                        "Endtoend_radius#plen1000#numstep1000000#time"]),
        rownum=1,
        colnum=1,

        picname="Rplot",
        freq_log=False,
        save_opt=False

):
    polylen = np.array([200,400,600,800,1000])
    txtnum = len(polylen)
    arr = []
    for i in range(txtnum):
        print(fname[i])
        arr.append(np.loadtxt(fpath + fname[i] + ".txt", dtype=int))
    arr = np.array(arr)
    print(arr.shape)
    print(arr[0].shape)
    nu = 0.6
    avalue = np.zeros(txtnum)
    varalue = np.zeros(txtnum)
    stdvalue = np.zeros(txtnum)
    for i in range(txtnum):
        avalue[i] = np.mean(arr[i][10000:],dtype=np.float64)
        stdvalue[i] = np.std(arr[i][10000:])
        #varalue[i] = np.var(arr[i][10000:])**0.5
    #print("Log[%f]}};" % (avalue[0] ** 0.5))

    #print(",{Log[%g], Log[%f]}};" % (polylen[txtnum - 1], avalue[txtnum - 1] ** 0.5))
    #i=0
    for i in range(txtnum):
        print("%g, %g, %g" % (polylen[i], avalue[i], stdvalue[i]))

    print("data = {{Log[%g], Log[%f]}"%(polylen[0], avalue[0]**0.5))
    for i in range(txtnum-3):
        print(",{Log[%g], Log[%f]}" % (polylen[i+1], avalue[i+1]**0.5))
    print(",{Log[%g], Log[%f]}};" % (polylen[txtnum- 1], avalue[txtnum - 1]**0.5))

    step = 10**6
    for j in range(9):

        print("%g *10^6:%g *10^6"%((j+1),(j+2)))
        print("data = {{Log[%g], Log[%f]}" % (polylen[0], np.mean(arr[0][(j+1)*step:(j+2)*step]) ** 0.5))
        for i in range(txtnum - 2):
            print(",{Log[%g], Log[%f]}" % (polylen[i + 1], np.mean(arr[i + 1][(j+1)*step:(j+2)*step]) ** 0.5))
        print(",{Log[%g], Log[%f]}};" % (polylen[txtnum - 1], np.mean(arr[txtnum - 1][(j+1)*step:(j+2)*step]) ** 0.5))


    
    '''fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    axxaxis = np.linspace(start=1,stop=len(arr[0]),num=len(arr[0]),endpoint=True)
    for i in range(txtnum):
        #arr_rescaled[i, 0:] = arr[i, 0:] / (200.0 + 200.0 * i)
        ax1.plot(axxaxis,arr[txtnum-1-i], alpha=1,label="N=%d" % (200 + 200 * (txtnum-1-i)))  #/ (200.0 + 200.0 * i)
    ax1.set_xlabel('Polymer configuration')  #, fontsize=16
    ax1.set_ylabel('R')  #, fontsize=16
    #ax1.set_title('', fontsize=18)
    ax1.legend(loc='upper right')
    if freq_log:
        ax1.set_xscale('log')
    plt.show()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300


    vararr = np.zeros((5,10))
    for i in range(txtnum):
        for j in range(10):
            vararr[i,j] = np.var(arr[i,0:100*(j+1)])**0.5

    fig = plt.figure() #figsize=(3.8, 2.8)
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2xaxis = np.linspace(start=100, stop=1000, num=10, endpoint=True)
    for i in range(txtnum):
        # arr_rescaled[i, 0:] = arr[i, 0:] / (200.0 + 200.0 * i)
        ax2.plot(ax2xaxis, vararr[i], alpha=1, label="N=%d" % (200 + 200 * (i)))  # / (200.0 + 200.0 * i)

    ax2.legend(loc='upper right')
    ax2.set_xscale('log')
    if freq_log:
        ax2.set_xscale('log')
    plt.show()
    '''
    return 0



#radevo(picname="20210329_MoeTw_1_1.png")


def Rhist(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210629test\\R_statistic\\overnight\\",
        fname=np.array([
                        "Endtoend_radius#plen200#numstep10000000#time",
                        "Endtoend_radius#plen400#numstep10000000#time",
                        "Endtoend_radius#plen600#numstep10000000#time",
                        "Endtoend_radius#plen800#numstep10000000#time",
                        "Endtoend_radius#plen1000#numstep10000000#time"
                        ]),
        flen=10000,
        rownum=1,
        colnum=2,
        binumber=100,
        xlimit=5,
        nu = 0.5926,
        lnk = 0.076,
        astart = 10000,
        picname="Rplot",
        freq_log=True,
        save_opt=False

):
    plen = np.array([200,400,600,800,1000])
    txtnum = len(fname)
    arr = []
    for i in range(txtnum):
        print(fname[i] + ".txt")
        arr.append(np.loadtxt(fpath + fname[i] + ".txt", dtype=int)[10000:]) #

    arr = np.array(arr)

    avalue = np.zeros(txtnum)
    varalue = np.zeros(txtnum)
    stdvalue = np.zeros(txtnum)
    #avalue = np.array([270.657])
    ''''''
    for i in range(txtnum):
        avalue[i] = np.log(np.mean(arr[i, :]))
        varalue[i] = np.var(np.log(arr[i, :]))
        stdvalue[i] = np.std(np.log(arr[i, :]))
    print("std of log")
    print(stdvalue)

    for i in range(txtnum):
        arr[i] = arr[i]**0.5/(plen[i]**(0.6))

    print(len(arr))
    print(len(arr[0]))
    print(arr[0,0])

    remax = np.amax(arr)
    histct = np.zeros((txtnum, binumber))
    #histacc = np.zeros((txtnum, binumber))

    for i in range(txtnum):
        histct[i], bin_edges = np.histogram(arr[i, :], bins=binumber, range=(0, remax), density=True)

    markers = np.array([".", "x", "D", "+", "^", "<", ">", "p","*","v","P","8","4","3","2","1"])

    fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.186, top=0.95, right=0.955,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax6 = fig.add_subplot(gs[0, 0])
    for i in range(txtnum):
        ax6.scatter(bin_edges[0:len(bin_edges)-1]+0.5*remax/binumber, histct[i], s=10, marker=markers[i], label="N=%d" % plen[i])
        #ax6.plot(bin_edges[1:len(bin_edges)], histct[i], label="N=%d" % (200 + 200 * i))
    ax6.set_xlim(-0.15,2.9)
    # ax6.set_xscale('log')
    ax6.set_ylabel('$f(ρ) = N^{ν} \dot P_N(R=ρN^{ν})$')  # f(\mathbf{ρ}) = N^{ν} \dot P_N(\mathbf{R}=\mathbf{ρ}N^{ν})
    ax6.set_xlabel('$ρ = R*N^{-ν}$')
    # ax6.set_title('N=200, 400, 600, 800, 1000')
    ax6.legend(loc='upper right',ncol=1)

    if save_opt:
        plt.savefig(fname="D:\\Mainz\\JGU\\Polymer\\Yuzhe figs\\Rhistogram-3D-b.png", format='png')
    #plt.show()

    fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax7= fig.add_subplot(gs[0, 0])
    ax7.errorbar(np.log(plen**2), avalue, yerr=stdvalue,fmt="o",markersize=2)  # s=30,
    #ax7.scatter(np.log(plen),avalue, s=30, marker="o")
    #-0.111651 + 0.639912 x
    ax7xaxis = np.linspace(start=-0.03,stop=14.33,num=1000,endpoint=True)
    ax7yaxis = lnk + nu*ax7xaxis
    ax7.text(0.2, 8.2, "$%g + %g ln(N^2) $"%(lnk,nu), color='green')
    ax7.plot(ax7xaxis, ax7yaxis, c='green')

    #ax7.set_xlim(-0.03, 7.41)
    #ax7.set_ylim(-0.04, 4.83)
    ax7.set_ylabel('$ln \langle R_N^2 \\rangle_{SAW} (10^5 average)$')
    ax7.set_xlabel('$ln (N^2)$')
    if save_opt:
        plt.savefig(fname="D:\\Mainz\\JGU\\Polymer\\Yuzhe figs\\Rhistogram-3D-a.png", format='png')
    plt.show()


    return 0


def intvhist(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210601test\\Overlap_interval_statistics\\",
        fname=np.array(["ovlpintv_statistic#plen200#dpthnum10000#","ovlpintv_statistic#plen400#dpthnum10000#", \
               "ovlpintv_statistic#plen600#dpthnum10000#","ovlpintv_statistic#plen800#dpthnum10000#", \
               "ovlpintv_statistic#plen1000#dpthnum10000#"]),
        flen = 10000,
        rownum=1,
        colnum=1,
        binumber=50,
        xlimit = 1.1,
        picname = "ovlpintv_statistic",
        freq_log=False,
        save_opt=False,
):
    txtnum = len(fname)
    arr = []
    for i in range(txtnum):
        print(fname[i])
        arr.append(np.loadtxt(fpath+fname[i]+".txt",dtype=float)[:flen])
    arr = np.array(arr)
    #print(arr)
    print(len(arr))
    print(len(arr[0]))

    nu=0.75

    arr_rescaled = np.zeros(arr.shape,dtype=float)

    for i in range(txtnum):
        arr_rescaled[i,0:] = arr[i,0:]/(200.0+200.0*i)

    remax = 1
    histct = np.zeros((txtnum, binumber))
    histacc = np.zeros((txtnum, binumber))

    for i in range(txtnum):
        histct[i], bin_edges = np.histogram(arr_rescaled[i,0:], bins=binumber, range=(0, remax))
    for i in range(txtnum):
        for j in range(len(histct[i])):
            histacc[i,j]=sum(histct[i,:j+1])

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights
    ax6 = fig.add_subplot(gs[0, 0])
    markers = np.array([".","x","D","+","^"])
    # = np.array([".", "x", "D", "+", "^"])
    for i in range(txtnum):
        ax6.scatter(bin_edges[1:len(bin_edges)], histacc[i]/flen,s=20,marker=markers[i],label="N=%d"%(200.0+200.0*i))
    ax6.set_xlim(0, xlimit)
    #ax6.set_xscale('log')
    ax6.set_ylabel('Counts')
    ax6.set_xlabel('I/N')
    #ax6.set_title('N=200, 400, 600, 800, 1000')
    ax6.legend(loc='lower right')
    plt.show()

    return 0

def pparahist(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210601test\\Pivotpara_Mtd_comparison\\",
        fname="pivotparamtdcomp#plen1000#numstep100000#time1622636360"
):

    arr = np.loadtxt(fpath+fname+".txt",dtype=float)

    print(len(arr))
    print(len(arr[0]))

    print(10**6*np.mean(arr[:,1]))
    print(10**6*np.mean(arr[:, 2]))
    print(10**6*np.mean(arr[:, 3]))

    return 0



def ovlpexmcplct_v2(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210601test\\Overlap_exam_comparison_v2\\",
        fname=np.array(["ovlpexmcomp_v2#plen200#numstep100000#time", "ovlpexmcomp_v2#plen400#numstep100000#time", \
                        "ovlpexmcomp_v2#plen600#numstep100000#time", "ovlpexmcomp_v2#plen800#numstep100000#time", \
                        "ovlpexmcomp_v2#plen1000#numstep100000#time"]),
        plen=np.linspace(start=200,stop=1000,num=5,endpoint=True)
):
    arr = []
    for i in range(5):
        print(fname[i])
        arr.append(np.loadtxt(fpath + fname[i] + ".txt", dtype=float))
    arr = np.array(arr)

    print(len(arr))
    print(arr[0].shape)

    avg=[]

    for i in range(5):
        print(fname[i])
        avg.append(np.mean(arr[i][:,4]))
    avg = np.array(avg)
    print(avg*2/(plen*(plen+1)))
    print(avg / (plen ))
    print("data = {{200, %g}, {400, %g}, {600, %g}, {800, %g}, {1000, %g}};" % (avg[0], avg[1], avg[2], avg[3], avg[4]))
    #-372.326 + 1.73085 x + 0.0102897 x^2
    fitx = np.linspace(start=200, stop=1000, num=5000, endpoint=True)
    fity = -372.326 + 1.73085 * fitx + 0.0102897* fitx**2

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(plen,avg,s=30,marker="x",color='green')
    ax.plot(fitx,fity)
    ax.set_xlabel("Number of atoms",fontsize=10)
    ax.set_ylabel("Step ($10^5$ average)",fontsize=10)
    ax.set_xlim(0,1100)
    ax.set_ylim(0, 13000)
    ax.text(100,8000,"$-372.326 + 1.73085 N + \dfrac{N^2}{97.184}$")
    plt.show()
    return 0

def ovlpexmcplct(
        fpath="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210601test\\Overlap_exam_comparison\\",
        fname=np.array(["ovlpexmcomp#plen200#numstep200000#time", "ovlpexmcomp#plen400#numstep200000#time", \
                        "ovlpexmcomp#plen600#numstep200000#time", "ovlpexmcomp#plen800#numstep200000#time", \
                        "ovlpexmcomp#plen1000#numstep200000#time"]),
        plen=np.linspace(start=200,stop=1000,num=5,endpoint=True)
):
    arr = []
    for i in range(5):
        print(fname[i])
        arr.append(np.loadtxt(fpath + fname[i] + ".txt", dtype=float))
    arr = np.array(arr)

    print(len(arr))
    print(arr[0].shape)

    avg=[]

    for i in range(5):
        print(fname[i])
        avg.append(np.mean(arr[i][:,4]))
    avg = np.array(avg)
    print(avg*2/(plen*(plen+1)))
    print(avg / (plen ))
    print("data = {{200, %g}, {400, %g}, {600, %g}, {800, %g}, {1000, %g}};" % (avg[0], avg[1], avg[2], avg[3], avg[4]))
    #6582.99- 36.994 x + 0.103577 x^2
    fitx = np.linspace(start=200, stop=1000, num=5000, endpoint=True)
    fity = 6582.99 + - 36.994 * fitx + 0.103577* fitx**2

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(plen,avg,s=30,marker="x",color='green')
    ax.plot(fitx,fity)
    ax.set_xlabel("Number of atoms",fontsize=10)
    ax.set_ylabel("Step ($2×10^5$ average)",fontsize=10)
    ax.set_xlim(0,1100)
    ax.set_ylim(0, 80000)
    ax.text(100,50000,"$6582.99 - 36.994 N + \dfrac{N^2}{9.655}$")
    plt.show()
    return 0

swopt = 0
if swopt ==0:
    Rplot()

if swopt==1:
    Rhist()

if swopt==3:
    intvhist()

if swopt==4:
    pparahist()

if swopt==5:
    ovlpexmcplct_v2()

if swopt==6:
    ovlpexmcplct()

if swopt==7:
    ovlpexmcplct()