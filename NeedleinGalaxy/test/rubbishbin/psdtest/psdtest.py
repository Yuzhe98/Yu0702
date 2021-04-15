import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append('C:\E\Yu0702\\NeedleinGalaxy')
import epa

a=np.random.rand(int(600))
b=np.linspace(start=0,stop=1,num=600)

fig = plt.figure()#figsize=(7, 9)
gs = gridspec.GridSpec(nrows=2, ncols=1)
spec_ax=fig.add_subplot(gs[0,0])
spec_w_ax=fig.add_subplot(gs[1,0])
c = spec_ax.plot(a)
c_w = spec_w_ax.plot(a+b)
spec_ax.set_ylabel('Amplitude [a.u.]', fontsize=12)
spec_ax.set_title('no drift', fontsize=12)
spec_w_ax.set_title('with drift', fontsize=12)
spec_w_ax.set_xlabel('Time [sec]', fontsize=12)
spec_w_ax.set_ylabel('Amplitude [a.u.]', fontsize=12)
spec_ax.set_ylim(0,2)
spec_w_ax.set_ylim(0,2)
fig.tight_layout()
plt.show()
