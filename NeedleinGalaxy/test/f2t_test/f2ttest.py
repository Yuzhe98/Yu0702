import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
import f2t
import epa
f2t.ftgenerator(
    samp_rate=512,  # sampling rate
    t_len=1,  # length of measuring time
    BG0_amp = 50,

    BG1_amp = 50,
    BG1_shape = -0.14,

    BG2_amp = 0.00,
    BG2_loc = 50,
    BG2_shape = 0.009,

    Gz_sigma = 3,
    Gz_mu = 0,

    Lz0_amp=30.0,
    Lz0_pkloc=50,
    Lz0_FWHM=0.09,

    Lz1_amp=0.0,
    Lz1_pkloc=25,
    Lz1_FWHM=0.09,

    Lz2_amp=0.0,
    Lz2_pkloc=100,
    Lz2_FWHM=0.09,
    phase_distribution='Uniform',

    retopt=False,  # choose whether to output
    store_path="C:" + "\\"+"E\Yu0702\\20190716\\",
    prefix='',
    suffix='',
    store_opt_f_amp=False,
    store_opt_f_AC=False,
    store_opt_t_A=False,
    make_plot=True,# Condition to produce plots
    Freqlog=False,
    verbose=False,  # Print details
    prpr=False
    )