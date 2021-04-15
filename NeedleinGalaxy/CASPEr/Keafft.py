import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, Model
from scipy import signal

import matplotlib.gridspec as gridspec

filepath = 'D:\\Mainz\\CASPEr\\Data\\'

def Lorentzian(
        x,
        center,
        gamma,
        A,
        offset
):
	# The line shape we'll fit to the data
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))

# Load the data, change the path and filename to the file you want to fit

def keafft(
        fpath='D:\\Mainz\\CASPEr\\Data\\',
        fstart=14,
        fend = 20,
        dwellt = 10*10**(-6),  # dwell time in second
        freq_opt=True,
        hist_opt=False,
        perc_opt=False,
        verbose = False
):
    num = fend-fstart+1
    data0 = np.loadtxt(filepath+'%d/data.csv'%fstart, unpack=True, delimiter=',')
    datalen = len(data0[0])
    #print(data.shape)
    del data0

    datarr = np.zeros((3*(fend-fstart+1), datalen))
    print(datarr[0:3,:].shape)

    for fi in np.arange(start=fstart, stop=fend+1, step=1):
        i= (fi-fstart)
        datarr[3*i:3*(i+1),:] = np.loadtxt(filepath + '%d/data.csv' % fi, unpack=True, delimiter=',')
    print(datarr.shape)

    stdv70 = np.zeros(num)
    stdv80 = np.zeros(num)
    stdv90 = np.zeros(num)
    stdv99 = np.zeros(num)
    stdv100 = np.zeros(num)

    stdv = np.zeros(num)
    Narr = np.array([1, 10, 100, 1000, 10000, 100000, 707976])
    for fi in np.arange(start=fstart, stop=fend + 1, step=1):
        i = (fi - fstart)
        print('N=%d'%(Narr[i]))
        FIDComplex = datarr[3*i+1] + 1j * datarr[3*i+2]
        FIDabs = np.abs(FIDComplex)
        Psum = np.vdot(FIDabs,FIDabs) / 2**16
        print('Psum=%g'%Psum)

        #ttt = 0.001*np.arange(2**16)
        #FIDComplex = np.sin(50*ttt) + 1j * np.cos(50*ttt)
        # Perform the FFT
        spectrum = np.fft.fft(FIDComplex)  #, norm='ortho'
        spec_abs = np.abs(spectrum)
        Psum_f = np.vdot(spec_abs, spec_abs)/ 100 * 100 / 2**32
        print('Psum_f=%g' % Psum_f)

        spectrum_ortho = np.fft.fft(FIDComplex, norm='ortho')  #
        spec_abs_2 = np.abs(spectrum_ortho)
        Psum_f_2 = np.vdot(spec_abs_2, spec_abs_2) / 2**16
        print('Psum_f-2=%g' % Psum_f_2)

        #binsize = 10**2 / 2**16
        #spectrum_ortho = np.abs(spectrum) /2**16   #

        frequencies = -np.fft.fftfreq(len(spectrum), d=dwellt * 1000)  # Set d to dwell time in ms

        #frequencies, spectrum = signal.welch(FIDComplex, 10**(5))  # , nperseg=1024
        stdv[i] = np.var(spectrum[1000:5000]) ** 0.5

        if freq_opt:
            fig = plt.figure()  # figsize=(10, 10)
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            freq_ax = fig.add_subplot(gs[0, 0])
            freq_ax.plot(frequencies, spec_abs/100)
            #freq_ax.plot(frequencies[1000:5000], spectrum[1000:5000], color='r')
            #print(np.var(spectrum[1000:5000]) ** 0.5)
            # freq_ax.set_ylim(0, maxcount * 1.1)
            freq_ax.set_ylabel('Amplitutude [ $\mu$V / $kHz$ ]')  # [V Hz^-1/2]
            freq_ax.set_xlabel('Freq [kHz]')
            freq_ax.set_title('Number of aquc = %d' % (Narr[i]))
            plt.show()

            fig = plt.figure()  # figsize=(10, 10)
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            freq_ax = fig.add_subplot(gs[0, 0])
            freq_ax.plot(frequencies, spec_abs_2)
            freq_ax.plot(frequencies[1000:5000], spec_abs_2[1000:5000], color='r')
            #print(np.var(spec_abs_2[1000:5000])**0.5)
            # freq_ax.set_ylim(0, maxcount * 1.1)
            freq_ax.set_ylabel('Amplitutude [ $\mu$V / $\sqrt{kHz}$ ]')  # [V Hz^-1/2]
            freq_ax.set_xlabel('Freq [kHz]')
            freq_ax.set_title('Number of aquc = %d'%(Narr[i]))



            #plt.savefig('Number of aquc = %d-2' % (10 ** i))
            plt.show()


        #print(frequencies[0:100])
        #print(frequencies[len(frequencies)//2:len(frequencies)//2+100])
        #anaa = np.sort(np.array([spectrum,frequencies]),axis=1)

        #print(np.var(spectrum[4000:8000]))
        #print(np.var(spectrum[8000:12000]))
        #plt.figure()
        #plt.plot(frequencies,spectrum)
        #plt.grid()
        #plt.show()

        #spec_sort = np.sort(spectrum[len(spectrum) // 3:len(spectrum) - len(spectrum) // 3])
        #if verbose: print(spec_sort[0:100])
        #len_sort = len(spec_sort)
        if hist_opt:
            fig = plt.figure()  # figsize=(10, 10)
            gs = gridspec.GridSpec(nrows=1, ncols=2)  #


            spec_ax = fig.add_subplot(gs[0, 0])
            spec_ax.plot(  frequencies,  # [len(spectrum) // 3:len(spectrum) - len(spectrum) // 3]
                         spectrum)
            spec_ax.set_ylabel('Amplitutude')
            spec_ax.set_xlabel('Freq [kHz]')
            spec_ax.set_title('')

            hist90_ax = fig.add_subplot(gs[0, 1])
            hist90_ax.hist(spec_sort[0:len_sort - len_sort // 100], bins=1000)
            hist90_ax.hist(spec_sort[0:len_sort - len_sort // 20], bins=1000)
            hist90_ax.hist(spec_sort[0:len_sort - len_sort // 10], bins=500)
            hist90_ax.hist(spec_sort[0:len_sort - 2 * len_sort // 10], bins=500)
            hist90_ax.hist(spec_sort[0:len_sort - 3 * len_sort // 10], bins=500)
            hist90_ax.legend(('99%', '95%', '90%', '80%', '70%'))
            # freq_ax.set_ylim(0, maxcount * 1.1)
            hist90_ax.set_ylabel('count')
            hist90_ax.set_xlabel('Amplitutude')
            hist90_ax.set_title('Histogram')

            plt.show()


        #print('Standard deviation = %g' % (np.var(spec_sort[0:len(spectrum)-len(spectrum)//10])))
        if perc_opt:
            Nperc = 20
            percentarr = np.linspace(start=100, stop=100 - 1 * Nperc, num=Nperc)
            stdvarr = np.zeros(Nperc)
            for pc in range(20):
                stdvarr[pc] = np.var(spec_sort[0:len_sort - pc * len_sort // 100]) ** 0.5

            fig = plt.figure()  # figsize=(10, 10)
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #

            std_ax = fig.add_subplot(gs[0, 0])
            std_ax.plot(percentarr, stdvarr)
            std_ax.set_ylabel('Standard deviation')
            std_ax.set_xlabel('The fraction of data [per cent]')
            plt.show()
            # std_ax.set_title('Histogram')
            '''
            stdv100[i] = np.var(spec_sort[:]) ** 0.5
            stdv99[i] = np.var(spec_sort[0:len_sort - len_sort // 100]) ** 0.5
            stdv90[i] = np.var(spec_sort[0:len_sort - len_sort // 10]) ** 0.5
            stdv80[i] = np.var(spec_sort[0:len_sort - 2 * len_sort // 10]) ** 0.5
            stdv70[i] = np.var(spec_sort[0:len_sort - 3 * len_sort // 10]) ** 0.5
            '''


    fig = plt.figure()  # figsize=(10, 10)
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #

    std_ax = fig.add_subplot(gs[0, 0])
    std_ax.plot(Narr, stdv)
    std_ax.scatter(Narr, stdv,c='r')
    std_ax.set_title('')
    std_ax.set_ylabel('Standard deviation [ $\mu$V/$\sqrt{kHz}$ ]')  # [V Hz^-1/2]
    std_ax.set_xlabel('N')
    std_ax.set_yscale('log')
    std_ax.set_xscale('log')
    plt.show()

    #print('Standard deviation (70 per cent) = %g' % (stdv70))
    #print('Standard deviation (80 per cent) = %g' % (stdv80))
    #print('Standard deviation (90 per cent) = %g' % (stdv90))
    #print('Standard deviation (99 per cent) = %g' % (stdv99))
    #print('Standard deviation (100 per cent) = %g' % (stdv100))




keafft(freq_opt=True, hist_opt=False)




'''


# To slightly automate the fitting, we estimate the center frequency here
centerEstimate = frequencies[np.argmax(spectrum)]

# Do the actual fitting
fitModel = Model(Lorentzian)
result = fitModel.fit(spectrum, x=frequencies, center= centerEstimate, gamma=0.1, A=0.1, offset = 0)

# print the results and plot everything
print(result.fit_report())
result.plot()

# gamma equals FWHM for a Lorenztian
'''