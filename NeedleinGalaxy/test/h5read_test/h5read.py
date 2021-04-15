import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def srconvt(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1 / 60
):
    array_len = len(array)
    if array_len % samp_rate != 0:
        raise ValueError('len(array)%samp_rate != 0')
    # if (samp_rate//target_rate - samp_rate/target_rate) != 0:
    # print(samp_rate//target_rate - samp_rate/target_rate)
    # raise ValueError('samp_rate%target_rate != 0')
    interv = int(samp_rate // target_rate)
    r = np.zeros(array_len // interv)
    for i in range(array_len // interv):
        r[i] = np.average(array[i * interv:(i + 1) * interv])
    return r


def minavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1 / 60
):
    return srconvt(array, samp_rate, target_rate)


def secavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1
):
    return srconvt(array, samp_rate, target_rate)


def tensecavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1 / 10
):
    return srconvt(array, samp_rate, target_rate)


def timeptcount(
        starttime=1049,
        endtime=1139,
        endpoint=False
):
    delta = (endtime - starttime)
    hour = int(endtime / 100) - int(starttime / 100)
    mint = endtime % 100 - starttime % 100
    num = mint + 60 * hour
    return num + int(endpoint)


def timearray(
        start=1009,
        end=1159,
):
    ta = np.zeros(timeptcount(start, end, endpoint=True), dtype=int)
    min = int(start/100)
    sec = start - 100 * min
    j=0
    for i in range(len(ta)):
        if sec == 60:
            min += 1
            sec = 0
        ta[i] = 100 * min + sec
        sec += 1
    return ta

def h5process(
        station_name='Hefei01',
        curdir='C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\',  # current directory
        samp_rate=512,
        year=2019,
        month=1,
        day=1,  # np.array([21,31]),
        start=1009,
        end=1159,
        sec=18,
        suffix='.hdf5',
        avg_method='tensec',
        raw_plot_opt=False,
        fft_plot_opt=False,
        plot_opt=False,
        store_opt=False,
        storepath='C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test'
):
    samp_rate_avg = samp_rate
    if avg_method == 'tensec':
        samp_rate_avg = 1 / 10
    os.chdir(curdir)  # + '\\%04d\\%02d\\%02d' % (year, month, day)
    ta = timearray(start=start, end=end)
      # print(ta[0:10])
    record = np.zeros((2, 60 * samp_rate * len(ta)))
    for ita in range(len(ta)):
        f = h5py.File(station_name + '_%04d%02d%02d_%04d%02d' % (year, month, day, ta[ita], sec) + suffix, 'r')
        print(station_name + '_%04d%02d%02d_%04d%02d' % (year, month, day, ta[ita], sec) + suffix)
        MagneticFields = 'MagneticFields'
        MFlen = len(f[MagneticFields])
        SanityChannel = 'SanityChannel'
        SClen = len(f[SanityChannel])
        record[0][ita * MFlen: (ita + 1) * MFlen] = f[MagneticFields]
        for iSC in range(SClen):
            if f[SanityChannel][iSC] == 0:
                print("San check failed. Process terminated")
                return -1
        record[1][ita * SClen: (ita + 1) * SClen] = f[SanityChannel]

    '''process no averaged data'''
    tf, f_welch = signal.welch(record[0], samp_rate, nperseg=len(record[0]))  #
    '''process averaged data'''
    avg = tensecavg(array=record[0], samp_rate=samp_rate, target_rate=samp_rate_avg)
    tfavg, favg_welch = signal.welch(avg, samp_rate_avg, nperseg=len(avg))
    if fft_plot_opt:
        plt.figure()
        plt.plot(tf, f_welch,label='No Average')  # linestyle=':', marker='o',
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amplitude [a.u.]')
        plt.title('Station: ' + station_name + ' ' +
                  'Y/M/D: %04d/%02d/%02d Hour&Minute: %04d to %04d'
                  % (year, month, day, ta[0], ta[-1]) + '\nFFT')
        plt.legend()
        plt.show()

    tf_period = 1 / tf[1:] / 60
    f_welch_norm = f_welch[1:] / max(f_welch[1:])

    tfavg_period = 1 / tfavg[1:] / 60
    favg_welch_norm = favg_welch[1:] / max(favg_welch[1:])

    if store_opt:
        os.chdir(storepath)
        fname0 = "Station_" + station_name + "_" + \
                 "%04d%02d%02d_%04d_to_%04d" % (year, month, day, ta[0], ta[-1]) + \
                 "_FFT_f_welch" + ".txt"
        #np.savetxt(fname0, np.array([tf, f_welch], dtype=np.float))  #tfavg_period np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]], dtype=np.float)

        fname1 = "Station_" + station_name + "_" + \
                 "%04d%02d%02d_%04d_to_%04d" % (year, month, day, ta[0], ta[-1]) + \
                 "_FFT_favg_welch" + ".txt"
        #np.savetxt(fname1, np.array([tfavg, favg_welch], dtype=np.float))#np.savetxt(fname1, favg_welch_norm)

    if plot_opt:
        plt.figure()
        # gs = gridspec.GridSpec(nrows=1, ncols=1)
        # raw_ax = fig.add_subplot(gs[0, 0])
        # sancheck_ax = fig.add_subplot(gs[0, 0])
        t = np.linspace(start=0, stop=len(record[0]) / samp_rate, num=len(record[0]), endpoint=False)
        plt.plot(t, record[0], label='SanityChannel True')  # , linestyle='--', color='b'
        # plt.plot(None, None, label='SanityChannel False',c='r)  # , linestyle='--', color='b'
        '''
        for i in range(len(record[0])):
            if 1 - record[1][i]:
                print("san check")
                raw_ax.plot(t[i], record[0][i], color='red', marker='x', linewidth='12')
        '''
        plt.ylabel('MagneticFields [pT]')
        plt.xlabel('time [s]')
        plt.title('Station: ' + station_name + ' ' +
                  'Time: ' + '%04d%02d%02d %04d to %04d'
                  % (year, month, day, ta[0], ta[-1]) + '\nRaw Data')
        plt.legend()
        plt.show()

        '''
        dont delete
        pyplot.plot(x,y,linewidth=, linestyle=, marker=, markevery=[])  markevery是要出现标志的位置的点，一般会是一个list
        '''
        '''
        fig1 = plt.figure()
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        welch_ax = fig1.add_subplot(gs[0, 0])
        welch_ax.plot(tf, f_welch / max(f_welch), color='b', linestyle='-', marker='o', label='No Average')  #
        #welch_ax.plot(tf_avg, f_avg / max(f_avg), color='g', linestyle='dashed', marker='x', label='Ten-second Average')  #
        welch_ax.set_xlabel('Frequency [Hz]')
        welch_ax.set_ylabel('Amplitude [a.u.]')
        plt.show()
        '''
        plt.figure()
        plt.plot(tf_period, f_welch_norm, color='b',
                 label='No Average')  # linestyle=':', marker='o',

        print('2ok')
        plt.plot(tfavg_period, favg_welch_norm + 0.1, color='g',
                 label='Ten-second Average')  # linestyle='-', marker='x',
        plt.xlabel('Period [min]')
        plt.ylabel('Amplitude [a.u.]')
        plt.title('Station: ' + station_name + ' ' +
                  'Time: ' + '%04d%02d%02d %04d to %04d'
                  % (year, month, day, ta[0], ta[-1]) + '\nFFT')
        plt.legend()
        plt.show()

        '''
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(nrows=2, ncols=1)
        # all_ax = fig.add_subplot(gs[:, :])
        # all_ax.set_title()
        # plt.grid(True)

        welch_ax = fig.add_subplot(gs[0, 0])
        welch_ax.plot(tf, f_welch)  #
        welch_ax.set_xlabel('Frequency [Hz]')
        welch_ax.set_ylabel('Amplitude')
        welch_ax.set_title("No Average")

        avg_ax = fig.add_subplot(gs[1, 0])
        avg_ax.plot(tf_avg, f_avg)  #
        avg_ax.set_xlabel('Frequency [Hz]')
        avg_ax.set_ylabel('Amplitude')
        avg_ax.set_title('Ten-second Average')  # , fontsize=22
        plt.show()
        '''
    return 0
def MultiDisplay(
        directory = 'C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\20191126',
        fname = np.array(['Station_Hefei01_20190107_0000_to_0919','Station_Hefei01_20190111_0000_to_0756']),
        verbose = False
):
    os.chdir(directory)
    colors = np.array(['r','g', 'b', 'k', 'm'])
    '''
    b: blue
　　g: green
　　r: red
　　c: cyan
　　m: magenta
　　y: yellow
　　k: black
　　w: white
    '''
      # print(os.getcwd())
    plt.figure()

    '''
    tf_period = 1 / tf[1:] / 60
    welch_norm = f_welch[1:] / max(f_welch[1:])
    tfavg_period = 1 / tfavg[1:] / 60
    avg_norm = f_avg[1:] / max(f_avg[1:])
    '''
    for i in range(len(fname)):
        if verbose:
            print("loading " + fname[i] + "\n")
        arrayrec = np.loadtxt(fname[i]+'.txt')
        if verbose:
            print(fname[i] + " successfully loaded\n")
        plt.plot(1/arrayrec[0][1:]/60, arrayrec[1][1:]/max(arrayrec[1][1:]) , c=colors[i], label=fname[i])  # linestyle='-', marker='x',
    #plt.xlabel('Freq [Hz]')
    plt.xlabel('Period [min]')
    plt.ylabel('Amplitude [a.u.]')
    plt.title('FFT on ten sec averaged data')
    plt.legend()
    plt.show()
    return 0


'''
h5process(
        station_name='beijing01',
        curdir=beijing01dir,  # current directory
        samp_rate=512,
        year=2019,
        month=1,
        day=26,  # np.array([21,31]),
        start=0,
        end=1200,
        sec=10,
        suffix='.hdf5',
        avg_method='tensec',
        plot_opt=False,
        store_opt=True,
        storepath=saveway
)'''


