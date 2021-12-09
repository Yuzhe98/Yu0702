import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.gridspec as gridspec
import NMRanalysis as ps
# Methanol density under 0.1 MPa / mol/L
# 20 C 24.687
# 0 C 25.271
# -20 C 25.860
# -90 C 28.00
methanoldensities = np.array([[20,24.687],[0,25.271],[-20,25.680],[-90,28]])
field = 500e-4
phi=np.zeros((4,2))
for i in range(len(methanoldensities)):
    phi[i,0] = ps.pulsedNMRsignalamp(
        mu0=1.25663706212e-6,  # in {\henry.\metre^{-1}
        mup=1.41060679736e-26,
        temp=methanoldensities[i,0],
        tempunit='C',
        sampvol=1.4e-3,  # in 'L'
        sampdens=methanoldensities[i,1],  # in mol/L
        NA=6.023e23,
        Bfield=field,  # in T
        SQUID_num='C6L1W',  # 'C73L1'
        L=953e-9,  # in H
        h=4.135667696 * 10 ** (-15),  #
        gyroratio=42.577478518 * 10 ** 6,  # in Hz/T
        k=8.617333262145 * 10 ** (-5),
        verbose=False,
    )
    print('temperature = %.6g C'%methanoldensities[i,0])
    print('sampdens = %.6g C' % methanoldensities[i, 1])
    print('C6L1W phi value = %.4g\n\n\n'%(1e6*phi[i,0]))

    phi[i,1] = ps.pulsedNMRsignalamp(
        mu0=1.25663706212e-6,  # in {\henry.\metre^{-1}
        mup=1.41060679736e-26,
        temp=methanoldensities[i, 0],
        tempunit='C',
        sampvol=1.4e-3,  # in 'L'
        sampdens=methanoldensities[i, 1],  # in mol/L
        NA=6.023e23,
        Bfield=field,  # in T
        SQUID_num='C73L1',  # 'C6L1W'
        L=953e-9,  # in H
        h=4.135667696 * 10 ** (-15),  #
        gyroratio=42.577478518 * 10 ** 6,  # in Hz/T
        k=8.617333262145 * 10 ** (-5),
        verbose=False,
    )
    print('temperature = %.6g C' % methanoldensities[i, 0])
    print('sampdens = %.6g C' % methanoldensities[i, 1])
    print('C73L1 phi value = %.4g\n\n\n' %(1e6*phi[i,1]))
print(1e6*phi.transpose())
#print(round(1e6*phi.transpose()[:],2))
result=1e6*phi.transpose()
print(result.shape)
print("C6L1W/$\sin{2\pi \\nu t}\ \mu\Phi_{0}$ & %.4g & %.4g & %.4g & %.4g \\\\ \n\
\midrule\n\
C73L1/$\sin{2\pi \\nu t}\ \mu\Phi_{0}$ & %.4g & %.4g & %.4g & %.4g \\\\"%(result[0,0],result[0,1],result[0,2],result[0,3],\
                                                                        result[1,0],result[1,1],result[1,2],result[1,3]))

























