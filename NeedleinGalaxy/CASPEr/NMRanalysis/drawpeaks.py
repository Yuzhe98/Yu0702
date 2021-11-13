import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

peakloc = np.array([7.13,7.12,7.10,7.05,6.94,6.87,5.89,4.81,4.79,4.77,4.73,4.58,4.56,4.55,4.51,4.39])
peakarea = np.array([1961.55, 2580.50,25278.05,151734.75,113214.12,5078.63,4935.94,2535.52,7482.10,16028.52,176545.36,927.19,1237.39,201243.44,103377.15,34517.24])

delta = 2.30821663
dd=201243.44#151734.75#201243.44
cshift =5.89-4.31
fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
ax0 = fig.add_subplot(gs[0, 0])
#for i in len(peakloc):
ax0.vlines([5.89,4.31],np.zeros(2),[max(peakarea)/dd,max(peakarea)/dd],colors='r',linestyles ="dashed")
#ax0.vlines([5.89,4.31],np.zeros(2),[max(peakarea)/dd,max(peakarea)/dd],colors='r',linestyles ="dashed")
ax0.vlines(peakloc,np.zeros(len(peakarea)),peakarea/dd)
ax0.vlines(peakloc,np.zeros(len(peakarea)),peakarea/dd)
ax0.hlines(125000/dd,7.05-delta,7.05,colors='g',linestyles ="dashed")
ax0.hlines(105000/dd,6.94-delta,6.94,colors='g',linestyles ="dashed")
ax0.hlines(20000/dd,7.10-delta,7.10,colors='g',linestyles ="dashed")
ax0.hlines(50000/dd,5.89-cshift,5.89,colors='b',linestyles ="dashed")
#ax0.hlines(50000/dd,8.39-cshift,8.39,colors='b',linestyles ="dashed")
#ax0.hlines(40000/dd,5.89+cshift,5.89,colors='b',linestyles ="dashed")
ax0.invert_xaxis()
ax0.set_xlabel('Peak location [ppm]')
ax0.set_ylabel('Area')
plt.grid()
plt.show()
