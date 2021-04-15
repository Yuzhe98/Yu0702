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
    array_len=len(array)
    if array_len%samp_rate!=0:
        raise ValueError('len(array)%samp_rate!=0')
    if (samp_rate//target_rate - samp_rate/target_rate)!=0:
        raise ValueError('samp_rate%target_rate!=0')
    interv = int(samp_rate//target_rate)
    r = np.zeros(array_len//interv)
    for i in range(array_len//interv):
        r[i] = np.average(array[i*interv:(i+1)*interv])
    return r
def minavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1 / 60
):
    return srconvt(array,samp_rate,target_rate)
def secavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1
):
    return srconvt(array,samp_rate,target_rate)
def tensecavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1/10
):
    return srconvt(array,samp_rate,target_rate)
station_name='Hefei01'
year = 2019
month = 1
day = 2
sec = 18
#  C:\E\Yu0702\NeedleinGalaxy\test\h5read_test\berkeley01\2019\01\01
os.chdir('C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\'+station_name+'\\%04d\\%02d\\%02d' % (year, month, day))

hour = np.arange(start=12, stop=10+1, step=1, dtype=int)
mint = np.arange(start=0, stop=59+1, step=1, dtype=int)
record = np.zeros(len(hour) * len(mint) * 60)
for h in range(len(hour)):
    for m in mint:
        f = h5py.File(station_name+'_%04d%02d%02d_%02d%02d%02d.hdf5' % (year, month, day, hour[h], m, sec), 'r')
        print(list(f.keys()))
        data_key = 'MagneticFields'
        dset = f[data_key]
        print("dset.shape=",dset.shape)
        print("dset.dtype=",dset.dtype)
        print(len(minavg(dset)))
        record[h * 3600 + m * 60: h * 3600 + m * 60 + 60] = minavg(dset)
        #record[(h) * 60 * 512:(h + 1) * 60 * 512] = dset
        '''
        fig = plt.figure()
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        welch_ax = fig.add_subplot(gs[0, 0])
        welch_ax.plot(record[0:(h + 1) * 60 * 512], color='red')  #
        plt.show()
        '''
#print(len([x for x in os.listdir('C:\E\Yu0702\\NeedleinGalaxy\\test\\201901\\01') if os.path.isfile(x)]))
tf,f_welch = signal.welch(record,1,nperseg=len(record))  #
fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)
welch_ax = fig.add_subplot(gs[0, 0])

welch_ax.plot(tf,f_welch)  #
welch_ax.set_xlabel('Frequency [Hz]',fontsize=22)
welch_ax.set_ylabel('Amplitude',fontsize=22)
welch_ax.set_title('Station: ' + station_name +'\n' +\
                   'Time: ' +'%04d.%02d.%02d %02d:00 to %02d:59' \
                   % (year, month, day, hour[0], hour[-1]), fontsize=22)
fig.tight_layout()
plt.show()

