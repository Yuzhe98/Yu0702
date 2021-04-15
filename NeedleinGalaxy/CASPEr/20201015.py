import numpy as np
import matplotlib.pyplot as plt
a=np.loadtxt('D:\Mainz\HIM_Lab_RecordsandNotes\\'+'1246_He_tail_base-Copy.txt')
#print(a)
fig,(f_amp_ax) = plt.subplots(1,1)

f_amp_ax.plot(a[:,1],a[:,0])  #
f_amp_ax.set_xlabel('Resistence [Ω]')
f_amp_ax.set_ylabel('Temperture [K]')

my_x_ticks = np.arange(1000,1300, 100)
#对比范围和名称的区别
#my_x_ticks = np.arange(-5, 2, 0.5)
my_y_ticks = np.arange(0, 330, 10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)


f_amp_ax.set_title('Temperture-Resistence Curve')

if True:
    #f_amp_ax.set_xlim(tf_full[1],)
    f_amp_ax.set_xscale('log')
    f_amp_ax.set_yscale('log')
#fig.tight_layout()
plt.grid()
plt.show()