import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def radevo(
        fname="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\failure_statistic#plen.txt",
        rownum=1,
        colnum=1,
        picname = "20200808_Moe_1_1",
        freq_log=True,
        save_opt=False,
        NotoSansCJKBold_path = "C:\\Users\zhenf\AppData\Local\Microsoft\Windows\Fonts\\NotoSansCJK-Bold.ttc",
        # C:\Users\zhenf\AppData\Local\Microsoft\Windows\Fonts\SourceHanSansCN-Bold.otf
        NotoSansCJKjpDemiLight_path = "C:\\Users\\zhenf\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NotoSansCJKjp-DemiLight.otf",
        RegularFont_path="C:\Windows\Fonts\SourceHanSansCN-Regular.otf", #
        BackupFont_path="C:\\Users\\zhenf\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NotoSansCJKjp-Regular.otf"
):
    '''

    '''
    arr = np.loadtxt(fname)
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights
    ax1 = fig.add_subplot(gs[0, 0])

    #ax1.plot(arr[:,1],arr[:,2])
    ax1.plot(arr[:, 1]/(arr[:, 0]+1))
    ax1.set_xlabel('Depth')  #, fontsize=16
    ax1.set_ylabel('Radius')  #, fontsize=16
    #ax1.set_title('', fontsize=18)
    if freq_log:
        ax1.set_xscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)
        ax1.set_yscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)
    plt.show()
    # fig.tight_layout()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300
    #plt.show()
    return 0



#radevo(picname="20210329_MoeTw_1_1.png")


def ovlplocst(
        fname="C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210419test\\ovlploc_statistic#plen-1000.txt",
        rownum=2,
        colnum=2,
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


    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights
    ax1 = fig.add_subplot(gs[0, 0])
    #ax1.plot(acc[:,1])
    #print(len(arr[:,1]))
    ax1.hist(arr[0:, 1], bins=10)
    ax1.set_xlabel('distance')  #, fontsize=16
    ax1.set_ylabel('counts')  #, fontsize=16
    #ax1.set_title('', fontsize=18)
    if freq_log:
        ax1.set_xscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)
        ax1.set_yscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)

    ax2 = fig.add_subplot(gs[0, 1])
    # ax1.plot(acc[:,1])
    #ax2.hist(arr[0:, 2], bins=100)
    ax2.hist(abs(arr[10000:, 3] - arr[10000:, 5])/abs(arr[10000:, 3] - arr[10000:, 4]), bins=200)  #
    ax2.set_xlabel('step num')  # , fontsize=16
    ax2.set_ylabel('counts')  # , fontsize=16

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(abs(arr[0:, 3]-arr[0:, 4]), bins=10000)
    ax3.set_xlabel('inum - pivotnum')  # , fontsize=16
    ax3.set_ylabel('counts')  # , fontsize=16

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(abs(arr[0:, 4] - arr[0:, 5]), bins=1000)
    ax3.set_xlabel('jnum - pivotnum')  # , fontsize=16
    ax3.set_ylabel('counts')  # , fontsize=16

    plt.show()
    # fig.tight_layout()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300
    #plt.show()

    print(np.average(arr[:,2]))

    return 0
'''
ovlplocst(
        fname="D:\\Mainz\\JGU\\Polymer\\Jingyan data\\R^2-N.txt",
        rownum=2,
        colnum=2,
        picname = "2020080",
        freq_log=False,
        save_opt=False,
)

#D:\Mainz\JGU\Polymer\Jingyan data

#C:\\Users\\zhenf\\source\\repos\\PolymerSimulation\\PolymerSimulation\\20210419test\\ovlploc_statistic#plen-1000 (5)
'''

def polyhist(
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
    binum=1000
    xlimit = 10.0
    arr_rescaled = np.zeros((500000))
    for i in range(5):
        arr_rescaled[i*100000:i*100000+100000] = arr[0:, i]/(200.0+200.0*i)**1.5

    #        arr_rescaled[i*100000:i*100000+100000] = arr[0:, 0]/(200.0)**1.5+arr[0:, 1]/(400.0)**1.5+arr[0:, 2]/(600.0)**1.5+arr[0:, 3]/(800.0)**1.5+arr[0:, 4]/(1000.0)**1.5

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=rownum, ncols=colnum)  # , width_ratios=widths, height_ratios=heights

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(arr[0:, 0]/(200.0)**1.5, bins=binum)
    ax0.set_xlim(0, xlimit)
    ax0.set_xlabel('end-to-end radius')  # , fontsize=16
    ax0.set_ylabel('counts')  # , fontsize=16

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(arr[0:, 1]/(400.0)**1.5, bins=binum)
    ax1.set_xlim(0, xlimit)
    ax1.set_xlabel('end-to-end radius')  #, fontsize=16
    ax1.set_ylabel('counts')  #, fontsize=16
    #ax1.set_title('', fontsize=18)


    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(arr[0:, 2]/(600.0)**1.5, bins=binum)
    ax2.set_xlim(0, xlimit)
    ax2.set_xlabel('end-to-end radius')  # , fontsize=16
    ax2.set_ylabel('counts')  # , fontsize=16
    # ax1.set_title('', fontsize=18)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(arr[0:, 3]/(800.0)**1.5, bins=binum)
    ax3.set_xlim(0, xlimit)
    ax3.set_xlabel('end-to-end radius')  #
    ax3.set_ylabel('counts')  #

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(arr[0:, 4]/(1000.0)**1.5, bins=binum)
    ax4.set_xlim(0,xlimit)
    ax4.set_xlabel('end-to-end radius')  #
    ax4.set_ylabel('counts')  #

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(arr_rescaled, bins=binum)
    ax5.set_xlim(0, xlimit)
    ax5.set_xlabel('end-to-end radius')  #
    ax5.set_ylabel('counts')  #

    plt.show()
    # fig.tight_layout()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300
    #plt.show()

    #print(np.average(arr[:,2]))

    return 0

polyhist(
        fname="D:\\Mainz\\JGU\\Polymer\\Jingyan data\\R^2-N.txt",
        rownum=2,
        colnum=3,
        picname = "2020080",
        freq_log=False,
        save_opt=False,
)