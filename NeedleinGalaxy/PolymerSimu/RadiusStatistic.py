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

    ax1.plot(arr[:,1],arr[:,2])
    ax1.set_xlabel('Depth', fontsize=16)
    ax1.set_ylabel('Radius', fontsize=16)
    ax1.set_title('', fontsize=18)
    #if freq_log:
        #ax1.set_yscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)
        #ax1.set_yscale('log')
        #ax1.set_ylim(f_lpass, f_hpass)
    plt.show()
    # fig.tight_layout()
    if save_opt:
        plt.savefig(fname=picname, format='png')  # , dpi=300
    #plt.show()
    return 0



radevo(

        picname="20210329_MoeTw_1_1.png")