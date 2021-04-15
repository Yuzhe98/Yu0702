import numpy as np
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
import multistat
import spacalcu

earth_radius=6371e3
Berkeley=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(-122.2572)/180])
Fribourg=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(7.1575)/180])
Hayward=np.array([earth_radius,np.pi*(90-46.7930)/180,np.pi*(-122.0540)/180])
Krakow=np.array([earth_radius,np.pi*(90-37.6564)/180,np.pi*(19.9046)/180])
Mainz=np.array([earth_radius,np.pi*(90-50.0286)/180,np.pi*(8.2346)/180])
Hefei=np.array([earth_radius,np.pi*(90-31.8428)/180,np.pi*(117.2525)/180])
locations=np.array([Berkeley,Fribourg,Hayward,Krakow,Mainz,Hefei])


def MultiStatTest0804(
    ref_path = 'C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
    fname = 'ref_ver04',
):
    ref = np.loadtxt(ref_path + fname + '.txt')
    #ansi=int(0.1*len(ref))
    tanda=np.ones(12)
    p=np.array([1,88*np.pi/180,2*np.pi/180.0])
    tanda[0::2]=spacalcu.DistInter(loc=locations,pert=p)
    multistat.QuickExam(
        ta=tanda,  # time interval and amp
        ref=ref,
        time_exm_opt=True,
        amp_exm_opt=False,
        exm_num=3,
        ans=p[1:],
        ans_opt=True,
        plot_opt=True,
        verbose_prog=True
    )



def MultiStatTest0805(
    ref_path = 'C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
    fname = 'ref_ver04',
):
    ref = np.loadtxt(ref_path + fname + '.txt')
    #ansi=int(0.1*len(ref))
    loop_num=100
    wro_rec=np.zeros(loop_num)
    std_rec=np.zeros(loop_num)
    for i in range(loop_num):
        tanda=np.ones(12)
        p=np.array([1,np.random.random_sample()+i/loop_num*np.pi,np.random.random_sample()+i/loop_num*2*np.pi])
        tanda[0::2]=spacalcu.DistInter(loc=locations,pert=p)
        print('i=',i)
        wro_rec[i],std_rec[i]=multistat.QuickExam(
            ta=tanda,  # time interval and amp
            ref=ref,
            time_exm_opt=True,
            amp_exm_opt=False,
            exm_num=3,
            ans=p[1:],
            ans_opt=True,
            plot_opt=False,
            verbose_prog=False
        )
    print('wrong time=',np.sum(wro_rec),'in ',loop_num)
    print('max std=', np.max(std_rec))
MultiStatTest0805()

'''
for i_la in np.linspace(start=30,stop=150,num=3,endpoint=True,dtype=int):
    for i_lo in np.linspace(start=0,stop=360,num=3,endpoint=False,dtype=int):
        MultiSensTest(ref_num=i_la*360+i_lo)
'''
'''
def MultiStatTest0804(
    ref_path = 'C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
    fname = 'ref_ver04',
    T=3600,
    sampling_rate = 512,  # sampling rate
    N_stat = 6,
    s_loc = np.array([[1800.57],[2000.9]]),
    confidcnum = int(2),
    ref_num=np.array([360*30+181,360*1200+271]),
    verbose_prog=True
):

    multistat.diramp(
        ta=r,  # time&amp record
        numsens=N_stat,
        confidc=confidc,
        ref_path='C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
        fname='ref_ver04',
        ans=ref_num,
        make_plot=True,
        colormap_name='plasma',
        colorbar_log=False,
        if_Lz=True,
        plot_FWHM=0.1,
        N_la=181,
        N_lo=360
    )

def MultiStatTest(
    ref_path = 'C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
    fname = 'ref_ver04',
    T=3600,
    sampling_rate = 512,  # sampling rate
    N_stat = 6,
    s_loc = np.array([[1800.57],[2000.9]]),
    confidcnum = int(2),
    ref_num=np.array([360*30+181,360*1200+271]),
    verbose_prog=True
):
    ref = np.loadtxt(ref_path + fname + '.txt')
    confidc = confidcnum * np.ones(N_stat, dtype=int)
    Lz_tloc = np.array([ref[ref_num[0]][2::2]])
    for i in range(len(ref_num)-1):
        Lz_tloc = np.append(Lz_tloc, np.array([ref[ref_num[i+1]][2::2]]), axis=0)
    Lz_tloc *= 1e+03
    Lz_tloc += s_loc
    if verbose_prog:print('Lz_tloc=', Lz_tloc)
    Lz_amp = np.array([ref[ref_num[0]][3::2]])
    #print('Lz_loc=', Lz_amp)
    for i in range(len(ref_num) - 1):
        Lz_amp = np.append(Lz_amp, np.array([ref[ref_num[i+1]][3::2]]), axis=0)
    Lz_amp **= 2
    Lz_amp *= 10
    if verbose_prog:print('Lz_amp=', Lz_amp)
    Lz_FWHM = 0.1 * np.array([[1., 1., 1., 1., 1., 1.0]])
    #print('Lz_FWHM=', Lz_FWHM)
    for i in range(len(ref_num) - 1):
        Lz_FWHM = np.append(Lz_FWHM, [0.1 * np.array([1., 1., 1., 1., 1., 1.0])], axis=0)

    if verbose_prog:print('Lz_FWHM=', Lz_FWHM)
    seglen = 2.0
    hipass = 5.0
    if verbose_prog:print('highpass=', hipass)

    n_freq = f2t.ftgenerator(
        samp_rate=512,  # sampling rate
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

    n_lin = np.linspace(start=0, stop=2, num=int(sampling_rate * T))
    n_combined = 2 * n_freq / np.average(np.abs(n_freq)) + 0.1 * n_lin
    t_combined = np.zeros((N_stat, sampling_rate * T), dtype=float)
    for i in range(N_stat):
        #tic = time.time()
        Lz = np.zeros(sampling_rate * T, dtype=float)
        for j in range(len(ref_num)):
            #print(Lz_FWHM[j][i])
            Lz += f2t.Lorentzian(
                starttime=0.0,
                samp_rate=512,  # sampling rate
                t_len=T,
                amp=Lz_amp[j][i],
                pkloc=Lz_tloc[j][i],
                FWHM=Lz_FWHM[j][i],
            )
        t_combined[i] = n_combined + Lz

    r = multistat.locating(
        s_serieses=t_combined,
        t_len=3600,
        samp_rate=512,
        seg_len=2,
        N_sens=6,
        hipass=5,
        confidc=confidc,
        prpr=False
    )

    multistat.diramp(
        ta=r,  # time&amp record
        numsens=N_stat,
        confidc=confidc,
        ref_path='C:\E\Yu0702\\NeedleinGalaxy\\test\\spacalcu_test\\',
        fname='ref_ver04',
        ans=ref_num,
        make_plot=True,
        colormap_name='plasma',
        colorbar_log=False,
        if_Lz=True,
        plot_FWHM=0.1,
        N_la=181,
        N_lo=360
    )
'''

