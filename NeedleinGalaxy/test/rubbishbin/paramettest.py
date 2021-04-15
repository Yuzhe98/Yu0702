import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import psdy
import ngenerator
def paratest():
    path='C:\E\Yu0702\pYUackages\\testdata\\'
    T=3600
    sampling_rate=512  # sampling rate
    N_noise=1
    N_seglen=10
    N_w=5
    N_f=2
    N=int(N_seglen*N_w*N_f)
    seglen=np.linspace(start=2,stop=20,num=N_seglen,endpoint=True,dtype=float)
    print(seglen)
    EMAw=np.linspace(start=40,stop=120,num=N_w,endpoint=True,dtype=float)
    print(EMAw)
    EMAf=np.linspace(start=0.02,stop=0.4,num=N_f,endpoint=True,dtype=float)
    print(EMAf)

    rec=np.zeros((N_noise,N))
    s_loc=1800.5
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
        Lz=ngenerator.Lorentzian(
        starttime=0.0,
        samp_rate=512,  # sampling rate
        t_len=T,
        amp=0.71,
        pkloc=s_loc,
        FWHM=0.1,
        )
        lin=np.linspace(start=0,stop=2,num=int(sampling_rate*T))
        t_combined=s_noise+Lz+lin


        for segj in range(N_seglen):
            #seglength=seglen[segj]
            #print('seglength=',seglength)
            for winj in range(N_w):
                #win= EMAw[winj]
                for fj in range(N_f):
                    tic=time.time()

                    print('segment length=',seglen[segj])
                    print('EMA window length=', EMAw[winj])
                    print('EMA factor=', EMAf[fj])
                    rec[i][(N_w*N_f)*segj+N_f*winj+fj]\
                        =psdy.excess_power(t_data=t_combined,  # signal series in time domain
                                    samp_rate=512,  # sampling rate
                                    t_len=T,  #length of measuring time
                                    seg_len=seglen[segj],  # Length of each seg in seconds
                                    seg_strd=seglen[segj]/2,  # overlap length, sgement stride in seconds
                                    avg_opt=True,
                                    avg_method='expmovmed',  #
                                    EMAwin=EMAw[winj],  #100
                                    EMAfcor=EMAf[fj],  #0.1+0.1*j
                                    make_plot=False,  # Condition to produce plots
                                    colorbar_log=True,
                                    retopt=True)
                    toc=time.time()
                    #tictoc=toc-tic
                    N_remain=N_noise*N-(i*N+(N_w*N_f)*segj+N_f*winj+fj+1)
                    t_remain=(toc-tic)*N_remain
                    hour=t_remain//3600
                    min=(t_remain-3600*hour)//60
                    second=t_remain-3600*hour-60*min
                    print("remaining time=",hour,':',min,':',second,'\n\n')
    loopend=time.time()
    print("loop time=",loopend -loopstart,'s')
    #numpy.savetxt(store_path+ "whitened" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + '.txt',f_welch)
    np.savetxt(path+"%g rec, n%g, s%g, w%g, f%g" % (loopend -loopstart,N_noise,N_seglen,N_w,N_f) + '.txt',rec)