import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
import powspec
import f2t
from scipy import signal

def Lztest(
    T=2,
    samp_rate=200,
    fwhm = 0.1
):
    time=np.linspace(start=0,stop=T,num=T*samp_rate)
    Lz01 = f2t.Lorentzian(
        starttime=0.0,
        samp_rate=samp_rate,  # sampling rate
        t_len=T,
        amp=1.0,
        pkloc=0.5*T,
        FWHM=fwhm,
    )
    s_noise01 = f2t.ftgenerator(
        samp_rate=samp_rate,  # sampling rate
        t_len=T,  # length of measuring time
        BG0_amp=500,
        BG1_amp=500,
        BG1_shape=-0.14,
        BG2_amp=0.00,
        BG2_loc=50,
        BG2_shape=0.009,

        Gz_sigma=40,
        Gz_mu=0,

        Lz0_amp=0.0,
        Lz1_amp=0.0,
        Lz2_amp=0.0,
        retopt=True,  # choose whether to output
        store_opt_f_amp=False,
        store_opt_f_AC=False,
        store_opt_t_A=False,
        make_plot=False,  # Condition to produce plots
        Freqlog=False,
        verbose=False,  # Print details
        prpr=False
    )
    Lz001 = f2t.Lorentzian(
        starttime=0.0,
        samp_rate=samp_rate,  # sampling rate
        t_len=T,
        amp=1.0,
        pkloc=0.5 * T,
        FWHM=0.1*fwhm,
    )
    s_noise001 = f2t.ftgenerator(
        samp_rate=samp_rate,  # sampling rate
        t_len=T,  # length of measuring time
        BG0_amp=500,
        BG1_amp=500,
        BG1_shape=-0.14,
        BG2_amp=0.00,
        BG2_loc=50,
        BG2_shape=0.009,

        Gz_sigma=40,
        Gz_mu=0,

        Lz0_amp=0.0,
        Lz1_amp=0.0,
        Lz2_amp=0.0,
        retopt=True,  # choose whether to output
        store_opt_f_amp=False,
        store_opt_f_AC=False,
        store_opt_t_A=False,
        make_plot=False,  # Condition to produce plots
        Freqlog=False,
        verbose=False,  # Print details
        prpr=False
    )
    time10 = np.linspace(start=0, stop=T, num=10*T * samp_rate)
    Lz0001 = f2t.Lorentzian(
        starttime=0.0,
        samp_rate=10*samp_rate,  # sampling rate50*200
        t_len=T,
        amp=10.0,
        pkloc=0.5 * T,
        FWHM=0.01 * fwhm,
    )
    s_noise0001 = f2t.ftgenerator(
        samp_rate=10*samp_rate,  # sampling rate
        t_len=T,  # length of measuring time
        BG0_amp=500,
        BG1_amp=500,
        BG1_shape=-0.14,
        BG2_amp=0.00,
        BG2_loc=50,
        BG2_shape=0.009,

        Gz_sigma=40,
        Gz_mu=0,

        Lz0_amp=0.0,
        Lz1_amp=0.0,
        Lz2_amp=0.0,
        retopt=True,  # choose whether to output
        store_opt_f_amp=False,
        store_opt_f_AC=False,
        store_opt_t_A=False,
        make_plot=False,  # Condition to produce plots
        Freqlog=False,
        verbose=False,  # Print details
        prpr=False
    )
    s20=f2t.SineEDec(
        starttime=0.0,
        samp_rate=samp_rate,  # sampling rate
        t_len=T,
        amp=1,
        Elambda=3,
        freq=20,
        phase0=0,
        startloc=0.5*T
    )
    s_noise20 = f2t.ftgenerator(
        samp_rate=samp_rate,  # sampling rate
        t_len=T,  # length of measuring time
        BG0_amp=500,
        BG1_amp=500,
        BG1_shape=-0.14,
        BG2_amp=0.00,
        BG2_loc=50,
        BG2_shape=0.009,

        Gz_sigma=40,
        Gz_mu=0,

        Lz0_amp=0.0,
        Lz1_amp=0.0,
        Lz2_amp=0.0,
        retopt=True,  # choose whether to output
        store_opt_f_amp=False,
        store_opt_f_AC=False,
        store_opt_t_A=False,
        make_plot=False,  # Condition to produce plots
        Freqlog=False,
        verbose=False,  # Print details
        prpr=False
    )
    tf01,wm01=signal.welch(Lz01+0.001*s_noise01, samp_rate, nperseg=samp_rate)
    tf001, wm001 = signal.welch(Lz001 + 0.001 * s_noise001, samp_rate, nperseg=samp_rate)
    tf0001, wm0001 = signal.welch(Lz0001 + 0.001 * s_noise0001, 20*samp_rate, nperseg=samp_rate)
    tf20, wm20 = signal.welch(s20 + 0.001 * s_noise20, samp_rate, nperseg=samp_rate)
    fig = plt.figure()
    row=4
    ax1 = fig.add_subplot(row, 2, 1)
    ax2 = fig.add_subplot(row, 2, 2)
    ax3 = fig.add_subplot(row, 2, 3)
    ax4 = fig.add_subplot(row, 2, 4)
    ax5 = fig.add_subplot(row, 2, 5)
    ax6 = fig.add_subplot(row, 2, 6)
    ax7 = fig.add_subplot(row, 2, 7)
    ax8 = fig.add_subplot(row, 2, 8)
    ax1.plot(time,Lz01+0.001*s_noise01)
    ax1.annotate('FWHM=0.1s', xy=(0.5*T, 0.5), xytext=(0.7*T, 0.6), fontsize=16,\
                 arrowprops=dict(arrowstyle='->'))
    ax2.plot(tf01,wm01)
    ax3.plot(time, Lz001 + 0.001 * s_noise001)
    ax3.annotate('FWHM=0.01s', xy=(0.5 * T, 0.5), xytext=(0.7 * T, 0.6), fontsize=16, \
                 arrowprops=dict(arrowstyle='->'))
    ax4.plot(tf001, wm001)
    ax5.plot(time10, Lz0001 + 0.001 * s_noise0001)
    ax5.annotate('FWHM=0.001s', xy=(0.5 * T, 5), xytext=(0.7 * T, 6), fontsize=16, \
                 arrowprops=dict(arrowstyle='->'))
    ax6.plot(tf0001, wm0001)

    ax7.plot(time, s20 + 0.001 * s_noise20)
    ax7.annotate('Frequency=20Hz', xy=(0.5 * T, 0.5), xytext=(0.7 * T, 0.6), fontsize=16, \
                 arrowprops=dict(arrowstyle='->'))
    ax8.plot(tf20, wm20)

    ax7.set_xlabel('time[s]', fontsize=18)
    ax8.set_xlabel('frequency[Hz]',fontsize=18)


    plt.show()

Lztest(
    T=2,
    samp_rate=200,
    fwhm = 0.1
)
def paratest():
    path='C:\E\Yu0702\\20190724\Lztest\FWHM=0.01s\\'
    T=3600
    sampling_rate=512  # sampling rate
    N_noise=1
    N1=7 #9
    N2=9#3
    N3=8#10
    N=int(N1*N2*N3)
    Lz_amp = np.linspace(start=0.8, stop=1.8, num=N1, endpoint=True, dtype=float)
    print('EMA window length=', Lz_amp)
    seglen=np.linspace(start=2,stop=10,num=N2,endpoint=True,dtype=float)
    print('segment length=',seglen)
    hipass=np.linspace(start=5, stop=40, num=N3, endpoint=True, dtype=float)
    print('highpass=', hipass)
    #EMAf=np.linspace(start=0.02,stop=0.12,num=N_3,endpoint=True,dtype=float)
    #print('EMA factor=',EMAf)

    rec=np.zeros((N_noise,N))
    s_loc=1800.5#signal location
    loopstart=time.time()
    for i in range(N_noise):
        s_noise=f2t.ftgenerator(
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
        fwhm=0.1
        Lz=f2t.Lorentzian(
        starttime=0.0,
        samp_rate=512,  # sampling rate
        t_len=T,
        amp=1.0,
        pkloc=s_loc,
        FWHM=fwhm,
        )
        lin=np.linspace(start=0,stop=2,num=int(sampling_rate*T))
        n_combined=1.0*s_noise/np.average(np.abs(s_noise))+1.0*lin

        for j1 in range(N1):
            t_combined=n_combined+Lz_amp[j1]*Lz
            for j2 in range(N2):
                for j3 in range(N3):
                    tic = time.time()
                    print('Lz_amp=', Lz_amp[j1])
                    print('Freq highpass=', hipass[j3])
                    print('segment length=', seglen[j2])
                    #print('EMA factor=', EMAf[j3])
                    f_whitened,aband\
                        =powspec.detect(t_data=t_combined,  # signal series in time domain
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
                    ''''''
                    print(axis0sum[seg_loc - aband - 3])
                    print(axis0sum[seg_loc - aband - 2])
                    print(axis0sum[seg_loc - aband - 1])
                    print(axis0sum[seg_loc - aband])
                    print(axis0sum[seg_loc - aband + 1])
                    print(axis0sum[seg_loc - aband + 2])
                    print(axis0sum[seg_loc - aband + 3])
                    print('threshold=', threshold)
                    print('events num=', len(count), ' in ', len(axis0sum))

                    print('error rate=',100*r,'%')

                    toc=time.time()

                    N_remain=N_noise*N-(i*N+(N2*N3)*j1+N3*j2+j3+1)
                    t_remain=(toc-tic)*N_remain
                    hour=int(t_remain//3600)
                    min=int((t_remain-3600*hour)//60)
                    second=t_remain-3600*hour-60*min
                    print("remaining time=", hour, ':', min, ':', second, '\n\n')
        np.savetxt(path + "20190724%g recFWHM=%gs, n%g, s_amp%g, seglen%g, hipass%g" % (toc-tic,fwhm, N_noise, N1, N2, N3) + '.txt',rec[i])
    loopend=time.time()
    print("loop time=",loopend -loopstart,'s')
    #numpy.savetxt(store_path+ "whitened" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + '.txt',f_welch)
