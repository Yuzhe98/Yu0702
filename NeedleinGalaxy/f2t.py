import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
def ExpDecSin(
    samp_rate=512,  # sampling rate. Default to 512 Hz
    t_len=10,  # siganl time length
    setend=False,
    amp=1,  # greatest value decaying amp curve
    Elambda=1,  # exponential factor, which is usually described as lamda
    freq=50,  # freq of sine wave
    phase0=0,  # initial phase. If set at pi/2, it would be a cosine wave
    startloc=5,  # the time location where the siganl appears

    store_opt=False,  # condition to store data. Default to False
    store_path='C:\E',  # store path
    prefix='201907',  # prefix for name of stored document
    suffix='',  # suffix for name of stored document
    ret_opt=True  # condition to return something
):
    data_num_bfo = int(startloc*samp_rate)  # number of data point before the signal airse
    data_num_aft = int(t_len*samp_rate)-data_num_bfo  # number of data point before the signal airse
    time = np.linspace(start=0,stop=t_len-startloc,num=data_num_aft,endpoint=setend,dtype=float)  # create the
    # signal series containing the injected signal

    SineExpDec = amp*np.exp(-1.0*Elambda*time)*np.sin(2.0*np.pi*freq*time+phase0)
    r = np.append(np.zeros(data_num_bfo,dtype=float),SineExpDec,axis=0)  #combine the series without signal with
    # the one containing signal
    if store_opt:  # store the signal series as txt
        fname = prefix + "ExpDecSine, t_len=%g, samp_rate=%gHz, amp=%g, startloc=%g" \
                % (t_len, samp_rate, amp, startloc) + suffix + '.txt'
        np.savetxt(store_path+fname, r)
    if ret_opt:  # return the signal series
        return r
    return 0

def Lorentzian(
    samp_rate=512,  # sampling rate. Default to 512 Hz
    t_len=10,  # signal time length
    setend=False,

    amp=1.0,  # amplitude of the lorentzian peak
    pkloc=5,  # the location of the peak
    FWHM=0.009,  # Full Width at Half Maximum

    store_opt=False,  # condition to store data. Default to False
    store_path='C:\E',  # store path
    prefix='201907',  # prefix for name of stored document
    suffix='',  # suffix for name of stored document
    ret_opt=True  # condition to return something. Default to True

):
    t = np.linspace(start=0, stop=t_len, num=t_len * samp_rate, endpoint=setend, dtype=float)
    #create the time series
    lz = amp / (1 + ((t - pkloc) / (0.5 * FWHM)) ** 2)  #create the signal series
    if store_opt:  # store the signal series as txt
        fname =prefix + "Lorentzian, t_len=%g, samp_rate=%gHz, amp=%g, pkloc=%g, FWHM=%g" \
               % (t_len, samp_rate,amp,pkloc,FWHM) + suffix + '.txt'
        np.savetxt(store_path+fname, lz)
    if ret_opt:  # return the signal series
        return lz
    return 0

def ftgenerator(
    samp_rate=512,  # sampling rate
    t_len=3600,  # length of time series
    # novel background
    BG0_amp = 500,  # amplitude
    # noise whose amplitude in freq domain has shape of exponentially decaying
    BG1_amp = 500,  # amplitude
    BG1_fctor = 0.14,  # decaying factor in Exp
    # noise whose amplitude in freq domain has shape of exponentially increasing
    BG2_amp = 0.00,  # amplitude
    BG2_loc = 50,  # start location in freq domain
    BG2_fctor = 0.009,   # decaying factor in Exp
    # Gaussian Noise
    Gz_sigma = 6,
    Gz_mu = 0,

    Lz0_amp=1.0,
    Lz0_pkloc=50,
    Lz0_FWHM=0.09,

    Lz1_amp=0.0,
    Lz1_pkloc=25,
    Lz1_FWHM=0.09,

    Lz2_amp=0.0,
    Lz2_pkloc=100,
    Lz2_FWHM=0.09,
    phase_distribution='Uniform',

    retopt=True,  # choose whether to output

    store_path="C:" + "\\"+"E\Yu0702\\20190716\\",
    prefix='',
    suffix='',
    store_opt_f_amp=True,
    store_opt_f_AC=False,
    store_opt_t_A=True,
    make_plot=False,# Condition to produce plots
    Freqlog=True,
    verbose=False,  # Print details
    prpr=False
    ):
    '''
    generate signal series from freq domain

    '''
    s_len = t_len * samp_rate
    s_len_half = int((t_len * samp_rate + 1) / 2)  # !!1be careful with the half length!!!
    # time series
    t = np.linspace(start=0.0, stop=t_len, num=s_len, endpoint=False, dtype=float)

    # time freq
    tf_half0 = np.linspace(start=0.0, stop=samp_rate / 2.0, num=s_len_half, endpoint=False, axis=0)
    tf_half1 = np.linspace(start=samp_rate / 2.0, stop=0.0, num=s_len_half, endpoint=False, axis=0)
    tf = np.append(tf_half0, tf_half1)
    tf_full = np.linspace(start=0.0, stop=samp_rate, num=s_len, endpoint=False, dtype=float)

    f_noise = np.zeros(s_len)
    f_noise[1:s_len_half + 1] = np.random.normal(loc=Gz_mu, scale=Gz_sigma, size=s_len_half)

    for i in np.arange(1, s_len_half):
        f_noise[s_len - i] = f_noise[i]

    f_amp = BG0_amp \
        + BG1_amp * np.exp(-1.0 * BG1_fctor * tf) \
        + BG2_amp * np.exp(BG2_fctor * (tf - BG2_loc)) \
        + Lz0_amp / (1 + ((tf - Lz0_pkloc) / (0.5 * Lz0_FWHM)) ** 2) \
        + Lz1_amp / (1 + ((tf - Lz1_pkloc) / (0.5 * Lz1_FWHM)) ** 2) \
        + Lz2_amp / (1 + ((tf - Lz2_pkloc) / (0.5 * Lz2_FWHM)) ** 2)\
        + f_noise
    f_amp = np.abs(f_amp)
    if store_opt_f_amp:
        fname = prefix + "noise f_domain_Amp" + ", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + suffix + '.txt'
        print(store_path+ fname)
        np.savetxt(store_path + fname, f_amp)
        #np.savetxt('C:\E\Yu0702\\20190711t_combined_raw,T=3600s,injection in middle,Freq=50Hz.txt', f_amp)

    if phase_distribution=='Uniform':
        f_phase = np.zeros(s_len)
        f_phase[1:s_len_half] = np.pi * 2 * np.random.random_sample(s_len_half - 1)
    else:
        raise ValueError('incorrect random distribution input')

    # create symmetry
    for i in np.arange(1, s_len_half):
        f_phase[s_len - i] = (-1) * f_phase[i]

    f_complexpart = np.cos(f_phase) + (0 + 1j) * np.sin(f_phase)

    f_AC = f_amp * f_complexpart  # both amp and complex part

    if store_opt_f_AC:
        fname = prefix + "noise f_domain_AmpComplex, phase distribution=" + phase_distribution +", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + suffix + '.txt'
        np.savetxt(store_path+fname, f_AC)

    t_A=np.real(ifft(f_AC))

    if store_opt_t_A:
        fname = prefix +"noise t_domain" +", phase distribution=" + phase_distribution +", T=%gs, samp_rate=%gHz" % (t_len, samp_rate) + suffix + '.txt'
        np.savetxt(store_path + fname, t_A)

    if make_plot:
        #
        fig,(f_amp_ax,f_phase_ax,t_A_ax) = plt.subplots(3,1)

        f_amp_ax.plot(tf_full,f_amp, color='red')  #
        f_amp_ax.set_xlabel('Frequency [Hz]')
        f_amp_ax.set_ylabel('Amplitude')

        f_amp_ax.set_title('Frequency-domain Amplitude')

        f_phase_ax.plot(tf_full,f_phase, color='b')#
        f_phase_ax.set_xlabel('Frequency [Hz]')
        f_phase_ax.set_ylabel('Amplitude')

        f_phase_ax.set_title('Phase')

        t_A_ax.plot(t,t_A, color='g')#
        t_A_ax.set_xlabel('Time [sec]')
        t_A_ax.set_ylabel('Amplitude')
        #if Freqlog: t_A_ax.set_yscale('log')
        t_A_ax.set_title('Time Domain Amplitude')

        if Freqlog:
            f_amp_ax.set_xlim(tf_full[1],)
            f_amp_ax.set_xscale('log')
        fig.tight_layout()
        plt.show()


    if retopt:
        return t_A
    return 0

