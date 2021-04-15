import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import psdy
import ngenerator
store_path='C:\E\Yu0702\\20190724\\amp,seglen,fctor\\'
#M=6
N_noise=5
N1 =7 #7
N2=4#4
N3=4#4
rec=np.zeros((N_noise,int(N1 * N2 * N3)))
for i in range(N_noise):
    rec[i]=np.loadtxt(store_path+'20190724Lz rec, n1, s4, w7, f4 (%d)'%(i)+'.txt').flatten()#
N_1plot=np.zeros(N1,dtype=float)
N_2plot=np.zeros(N2, dtype=float)
N_3plot=np.zeros(N3, dtype=float)
for i in range(N1):N_1plot[i]= np.sum(rec[:, (N2 * N3) * i:(N2 * N3) * (i + 1)]) / (N_noise * N2 * N3)
for i in range(N2):
    for j in range(N1):
        N_2plot[i]+= np.sum(rec[:, N3 * i + (N2 * N3) * j:N3 * i + (N2 * N3) * j + N3]) / (N_noise * N1 * N3)
for i in range(N3):N_3plot[i]= np.sum(rec[:, ::N3]) / (N_noise * N2 * N1)
'''
plt.plot(np.mean(rec[:,::],axis=0))
#plt.plot(rec[0][0::4])
plt.ylabel('error rate')
plt.grid()
plt.show()
'''
fig = plt.figure()
gs = gridspec.GridSpec(nrows=3, ncols=1)
oned0_ax = fig.add_subplot(gs[0, 0])
oned1_ax = fig.add_subplot(gs[1, 0])
oned2_ax = fig.add_subplot(gs[2, 0])

oned0_ax.plot(N_1plot)
oned1_ax.plot(N_2plot)
oned2_ax.plot(N_3plot)

fig.tight_layout()
plt.show()

'''
fig = plt.figure()
gs = gridspec.GridSpec(nrows=6, ncols=1)
oned0_ax = fig.add_subplot(gs[0, 0])
oned1_ax = fig.add_subplot(gs[1, 0])
oned2_ax = fig.add_subplot(gs[2, 0])
oned3_ax = fig.add_subplot(gs[3, 0])
oned4_ax = fig.add_subplot(gs[4, 0])
oned5_ax = fig.add_subplot(gs[5, 0])

oned0_ax.plot(rec[0])
oned1_ax.plot(rec[1])
oned2_ax.plot(rec[2])
oned3_ax.plot(rec[3])
oned4_ax.plot(rec[4])
oned5_ax.plot(rec[5])

fig.tight_layout()
plt.show()
'''