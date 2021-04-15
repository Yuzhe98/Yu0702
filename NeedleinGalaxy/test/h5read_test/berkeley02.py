import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
month = 1
date = 2
#C:\E\Yu0702\NeedleinGalaxy\test\h5read_test\berkeley01\2019\01\01
os.chdir('C:\E\Yu0702\\NeedleinGalaxy\\test\h5read_test\\berkeley01\\2019\\%02d\\%02d'%(month, date))
def minavg(
        array=np.zeros(513),
        samp_rate=512,
        target_rate=1
):
    array_len=len(array)
    if array_len%samp_rate!=0:
        raise ValueError('len(array)%samp_rate!=0')
    if samp_rate%target_rate!=0:
        raise ValueError('samp_rate%target_rate!=0')
    interv = samp_rate//target_rate
    rest = np.zeros(array_len//interv)
    for i in range(array_len//interv):
        rest[i] = np.average(array[i*interv:(i+1)*interv])
    return rest
station_name='berkeley01'

hour = np.linspace(start=0, stop=11, num=12, endpoint=True, dtype=int)
mint = np.linspace(start=0, stop=59, num=60, endpoint=True, dtype=int)
record = np.zeros(len(hour) * len(mint) * 60)
for h in range(len(hour)):
    for m in mint:
        #berkeley01_20190101_003440
        f = h5py.File('berkeley01_201901%02d_%02d%02d40.hdf5'%(date, hour[h], m), 'r')
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
tf,f_welch = signal.welch(record,1,nperseg=len(record)//2)  #
fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)
welch_ax = fig.add_subplot(gs[0, 0])
#tf[1:]=1/tf[1:]/60
welch_ax.plot(tf,f_welch)  #
welch_ax.set_xlabel('Frequency [Hz]',fontsize=18)
welch_ax.set_ylabel('Amplitude',fontsize=18)
welch_ax.set_title('Station: '+station_name+'\n'+'Time: '+'2019.01.%02d %02d:00 to %02d:59'%(date, hour[0], hour[-1]),fontsize=18)
fig.tight_layout()
plt.show()