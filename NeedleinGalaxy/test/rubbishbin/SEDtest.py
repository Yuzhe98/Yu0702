import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import psdy
import signaly
import ngenerator
def paratest():
    path='C:\E\Yu0702\\20190724\SEDtest\\2\\'
    T=3600
    sampling_rate=512  # sampling rate
    N_noise=1
    N1=6 #9
    N2=4#3
    N3=11#10
    N=int(N1*N2*N3)
    SED_amp = np.linspace(start=0.4, stop=1.4, num=N1, endpoint=True, dtype=float)
    print('SED_amp=', SED_amp)
    seglen=np.linspace(start=2,stop=14,num=N2,endpoint=True,dtype=float)
    print('segment length=',seglen)
    hipass=np.linspace(start=5, stop=55, num=N3, endpoint=True, dtype=float)
    print('highpass=', hipass)
    #EMAf=np.linspace(start=0.02,stop=0.12,num=N_3,endpoint=True,dtype=float)
    #print('EMA factor=',EMAf)

    rec=np.zeros((N_noise,N))
    s_loc=1800.5#signal location
    loopstart=time.time()
    for i in range(N_noise):
        s_noise=ngenerator.ftgenerator(
        samp_rate=512,  # sampling rate
        t_len=T,  # length of measuring time
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
        Freqlog=False,
        verbose=False,  # Print details
        prpr=False
        )
        Freq=5
        SED=ngenerator.SineEDec(
            starttime=0.0,
            samp_rate=512,  # sampling rate
            t_len=T,
            setend=False,
            amp=1.0,
            Elambda=0.3,
            freq=Freq,
            phase0=0,
            startloc=s_loc)
        lin=np.linspace(start=0,stop=2,num=int(sampling_rate*T))
        n_combined=1*s_noise/np.average(np.abs(s_noise))+1*lin

        for j1 in range(N1):
            t_combined=n_combined+SED_amp[j1]*SED
            for j2 in range(N2):
                for j3 in range(N3):
                    tic = time.time()
                    print('SED_amp=', SED_amp[j1])
                    print('Freq highpass=', hipass[j3])
                    print('segment length=', seglen[j2])
                    #print('EMA factor=', EMAf[j3])
                    f_whitened,aband\
                        =signaly.detect(t_data=t_combined,  # signal series in time domain
                                    samp_rate=512,  # sampling rate
                                    t_len=T,  #length of measuring time
                                    seg_len=seglen[j2],  # Length of each seg in seconds
                                    seg_strd=seglen[j2]/2,  # overlap length, sgement stride in seconds
                                    f_lpass=0,
                                    f_hpass=hipass[j3],
                                    avg_opt=True,
                                    avg_method='expmovmed',
                                    EMAwin=50*seglen[j2],  #100
                                    EMAfcor=0.02,  #0.1+0.1*j
                                    make_plot=False,  # Condition to produce plots
                                    colorbar_log=True,
                                    retopt=True)

                    bef = int(2*s_loc / seglen[j2])
                    if bef * seglen[j2]/2.0 <= s_loc and (bef + 1) * seglen[j2]/2.0 >= s_loc:
                        seg_loc = bef
                    else:
                        raise ValueError('23333333')
                    axis0sum = np.sum(f_whitened, axis=1)
                    threshold = max(axis0sum[seg_loc - aband - 2:seg_loc - aband + 1])
                    count = [x for x in axis0sum if x >= threshold]
                    r = (len(count) - 1.0) / (1.0 * len(axis0sum))
                    rec[i][(N2 * N3) * j1 + N3 * j2 + j3]=r
                    '''
                    print(axis0sum[seg_loc - aband - 3])
                    print(axis0sum[seg_loc - aband - 2])
                    print(axis0sum[seg_loc - aband - 1])
                    print(axis0sum[seg_loc - aband])
                    print(axis0sum[seg_loc - aband + 1])
                    print(axis0sum[seg_loc - aband + 2])
                    print(axis0sum[seg_loc - aband + 3])
                    print('threshold=', threshold)
                    print('events num=', len(count), ' in ', len(axis0sum))
                    '''
                    print('error rate=',100*r,'%')

                    toc=time.time()

                    N_remain=N_noise*N-(i*N+(N2*N3)*j1+N3*j2+j3+1)
                    t_remain=(toc-tic)*N_remain
                    hour=int(t_remain//3600)
                    min=int((t_remain-3600*hour)//60)
                    second=t_remain-3600*hour-60*min
                    print("remaining time=", hour, ':', min, ':', second, '\n\n')
        np.savetxt(path + "20190724%g SEDrec,freq%gHz n%g, s_amp%g, seglen%g, hipass%g" % (toc-tic, Freq,N_noise, N1, N2, N3) + '.txt',rec[i])
    loopend=time.time()
    print("loop time=",loopend -loopstart,'s')
    #numpy.savetxt(store_path+ "whitened" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + '.txt',f_welch)
