import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def polyhist_Xu(
        fname="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210419test\\ovlploc_statistic#plen-1000.txt",
        rownum=2,
        colnum=3,
        picname = "2020080",
        freq_log=False,
        save_opt=False,
):
    '''

    '''
    arr = np.loadtxt(fname)

    print(len(arr))
    print(len(arr[0]))
    #arr.reshape((5,100000))
    #print(len(arr))
    #print(len(arr[0]))
    binum=300
    xlimit = 3

    nu=0.75

    arr_rescaled = np.zeros(arr.shape)


    for i in range(5):
        arr_rescaled[0:,i] = arr[0:, i]**0.5/(200.0+200.0*i)**nu
    remax = np.amax(arr_rescaled)

    #        arr_rescaled[i*100000:i*100000+100000] = arr[0:, 0]/(200.0)**1.5+arr[0:, 1]/(400.0)**1.5+arr[0:, 2]/(600.0)**1.5+arr[0:, 3]/(800.0)**1.5+arr[0:, 4]/(1000.0)**1.5

    #numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    axarray = np.array([ax0, ax1, ax2, ax3, ax4, ax5])
    histacc = np.zeros((5,binum))

    for i in range(5):
        histacc[i], bin_edges = np.histogram(arr_rescaled[0:, i], bins=binum, range=(0, remax),density=True)
        # weights=(200.0 + 200.0 * i) ** nu / len(arr_rescaled[0:, i]) * np.ones(len(arr_rescaled[0:, i]))
        axarray[i].hist(arr_rescaled[0:,i], bins=binum,range=(0,remax), density=True)  #,weights=(200.0+200.0*i)**nu/len(arr_rescaled[0:,i])*np.ones(len(arr_rescaled[0:,i]))
        axarray[i].set_xlim(0, xlimit)
        axarray[i].set_ylabel('Density')  # *$N^{ν}$ ,fontsize=16
        axarray[i].set_xlabel('$R*N^{-ν}$')
        axarray[i].set_title('N=%d'%(200+200*i))

    #ax5.plot(bin_edges[:len(bin_edges)-1], np.sum(histacc, axis=0))
    for i in range(5):
        ax5.scatter(bin_edges[:len(bin_edges) - 1], histacc[i])
    #ax5.hist(arr_rescaled, bins=binum)
    ax5.set_xlim(0, xlimit)
    #ax5.set_xlabel('end-to-end radius')  #
    ax5.set_ylabel('Density')  #Σ *$N^{ν}$ (not normalized)
    ax5.set_xlabel('$R*N^{-ν}$')
    ax5.set_title('N=200, 400, 600, 800, 1000 accumulated')

    fig.tight_layout()
    plt.show()
    ''''''
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
    ax6 = fig.add_subplot(gs[0, 0])
    for i in range(5):
        ax6.scatter(bin_edges[:len(bin_edges) - 1], histacc[i])

    ax6.set_xlim(0, xlimit)
    ax6.set_ylabel('Density')
    ax6.set_xlabel('$R*N^{-ν}$')
    ax6.set_title('N=200, 400, 600, 800, 1000 accumulated')
    plt.show()

    return 0


def Rplot_Xu(
        fpath="D:\\Mainz\\JGU\\Polymer\\Jingyan data\\",
        fname="R^2-N",
        rownum=1,
        colnum=1,
        picname="Rplot-2D",
        freq_log=True,
        save_opt=True

):


    arr=np.loadtxt(fpath + fname + ".txt", dtype=float)#[0:flen+1])
    print(arr.shape)
    for i in range(5):

        rvalues = arr[1000:,i]**0.5
        ravg = np.mean(rvalues)
        rvar = np.var(rvalues)**0.5

        print(rvalues.shape)
        print(ravg)
        print(rvar)
        print(np.log(ravg))

    arr=arr.transpose(1, 0)
    arr = arr**0.5
    print(arr.shape)

    fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    axxaxis = np.linspace(start=1,stop=len(arr[0]),num=len(arr[0]),endpoint=True)
    for i in range(5):
        #arr_rescaled[i, 0:] = arr[i, 0:] / (200.0 + 200.0 * i)
        ax1.plot(axxaxis,arr[4-i], alpha=1,label="N=%d" % (200 + 200 * (4-i)))  #/ (200.0 + 200.0 * i)
    ax1.set_xlabel('Polymer configuration')  #, fontsize=16
    ax1.set_ylabel('R')  #, fontsize=16
    #ax1.set_title('', fontsize=18)
    ax1.legend(loc='upper right')
    if freq_log:
        ax1.set_xscale('log')
    plt.show()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300
    return 0

def Rhist_Xu(
        fpath="D:\\Mainz\\JGU\\Polymer\\Jingyan data\\",
        fname="R^2-N",
        rownum=1,
        colnum=2,
        binumber=100,
        astart = 1000,
        nu=0.75,
        picname="Rplot-2D",
        freq_log=True,
        save_opt=False

):
    arr=np.loadtxt(fpath + fname + ".txt", dtype=float)#[0:flen+1])
    print(arr.shape)

    arr=arr.transpose(1, 0)
    arr = arr**0.5
    print(arr.shape)

    axxaxis = np.linspace(start=1,stop=len(arr[0]),num=len(arr[0]),endpoint=True)

    arrnum = len(arr)
    avalue = np.zeros(arrnum)
    varalue = np.zeros(arrnum)
    for i in range(arrnum):
        avalue[i] = np.log(np.mean(arr[i, :]))
        varalue[i] = np.var(np.log(arr[i, :]))**0.5


    arr_rescaled = arr
    for i in range(arrnum):
        arr_rescaled[i] = arr_rescaled[i] / ((200.0 + 200.0 * i) ** nu)

    print(len(arr))
    print(len(arr[0]))
    print(arr[0, 0])

    print("data = {{Log[200], %g}, {Log[400], %g}, {Log[600], %g}, {Log[800], %g}, {Log[1000], %g}};" % (
    avalue[0], avalue[1], avalue[2], avalue[3], avalue[4]))
    print("vardata = {{Log[200], %g}, {Log[400], %g}, {Log[600], %g}, {Log[800], %g}, {Log[1000], %g}};" % (
        varalue[0], varalue[1], varalue[2], varalue[3], varalue[4]))

    remax = np.amax(arr)
    histct = np.zeros((arrnum, binumber))
    # histacc = np.zeros((arrnum, binumber))

    for i in range(arrnum):
        histct[i], bin_edges = np.histogram(arr_rescaled[i, astart:], bins=binumber, range=(0, remax), density=True)

    fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax6 = fig.add_subplot(gs[0, 0])
    markers = np.array([".", "x", "D", "+", "^"])

    for i in range(arrnum):
        ax6.scatter(bin_edges[0:len(bin_edges) - 1] + 0.5 * remax / binumber, histct[i], s=10, marker=markers[i],
                    label="N=%d" % (200.0 + 200.0 * i))


    ax6.set_ylabel('$f(ρ) = N^{ν} \dot P_N(R=ρN^{ν})$')
    ax6.set_xlabel('$ρ = R*N^{-ν}$')
    ax6.legend(loc='upper right')
    plt.savefig(fname="D:\\Mainz\\JGU\\Polymer\\Yuzhe figs\\Rhistogram-2D-b.png", format='png')
    plt.show()

    fig = plt.figure(figsize=(3.8, 2.8))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.193, top=0.95, right=0.967,
                        bottom=0.18, wspace=0.198, hspace=0.1)
    ax7 = fig.add_subplot(gs[0, 0])

    #plt.errorbar(a, b, yerr=c, fmt="o", color="r")
    ax7.errorbar(np.log(np.linspace(start=200, stop=1000, num=arrnum, endpoint=True)), avalue,yerr=varalue, fmt="o") #s=30,
    #-0.16966 + 0.752773 x
    ax7xaxis = np.linspace(start=-0.03, stop=7.33, num=1000, endpoint=True)
    ax7yaxis = -0.16966 + 0.752773 * ax7xaxis
    ax7.text(0.2, 4.5, "$-0.16966 + 0.752773 ln(N) $", color='green')
    ax7.plot(ax7xaxis, ax7yaxis, c='green')

    ax7.set_xlim(-0.03, 7.33)
    ax7.set_ylim(-0.04, 5.65)
    ax7.set_ylabel('$ln (R) (10^5 average)$')
    ax7.set_xlabel('$ln (N)$')
    plt.savefig(fname="D:\\Mainz\\JGU\\Polymer\\Yuzhe figs\\Rhistogram-2D-a.png", format='png')
    plt.show()

    return 0


swopt = 1
if swopt ==0:
    Rplot_Xu()

if swopt==1:
    Rhist_Xu()