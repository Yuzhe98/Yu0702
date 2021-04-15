import numpy as np
import sys
sys.path.append('C:\E\Yu0702\pYUackages')

import matplotlib.pyplot as plt
import psdy
import ngenerator
import time


T=3600

sampling_rate=512  # sampling rate
M=5
N=16
varrec=np.zeros((M,N))
for i in range(M):
    tic=time.time()
    s_noise=ngenerator.ftgenerator(
    samp_rate=512,  # sampling rate
    T=7200,  # length of measuring time
    BG0_amp = 500,
    BG1_amp = 500,
    BG1_shape = -0.14,
    BG2_amp = 0.00,
    BG2_loc = 50,
    BG2_shape = 0.009,

    Gz_sigma = 40,
    Gz_mu = 0,

    Lz0_amp=0.0,
    Lz1_amp=0.0,
    Lz2_amp=0.0,

    retopt=True,  # choose whether to output
    store_opt_f_amp=False,
    store_opt_f_AC=False,
    store_opt_t_A=False,
    make_plot=False,# Condition to produce plots
    Freqlog=True,
    verbose=False,  # Print details
    prpr=False
    )
    toc=time.time()
    print("generating time=",toc-tic,'s')

    for j in range(N):
        seglength=2.0+4*j
        print('seglength=',seglength)
        varrec[i][j]=psdy.excess_power(t_data=s_noise,  # signal series in time domain
                  samp_rate=512,  # sampling rate
                  t_len=T,  #length of measuring time
                  seg_len=10,  # Length of each seg in seconds
                  seg_strd=5,  # overlap length, sgement stride in seconds
                  avg_opt=True,
                  avg_method='expmovmed',  #
                  EMAwin=30+10*j,#100
                  EMAfcor=0.3,#0.1+0.1*j
                  make_plot=False,  # Condition to produce plots
                  retopt=True)
        print(varrec[i][j])
#print(varrec)
#plt.plot(np.mean(varrec,axis=0))
plt.plot(np.linspace(start=30,stop=30+N*10,num=N),np.mean(varrec,axis=0))
#plt.plot(np.linspace(start=2,stop=2+4*N,num=N),np.mean(varrec,axis=0))
#plt.plot(np.linspace(start=0.1,stop=0.1+N*0.1,num=N),np.mean(varrec,axis=0))
#plt.xlabel('EMA window length [s]')
#plt.xlabel('EMA factor')
plt.xlabel('Segment Length[s]')
plt.ylabel('variance')
plt.grid()
plt.show()