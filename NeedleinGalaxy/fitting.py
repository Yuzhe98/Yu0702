#最小二乘拟合实例
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def func(x=np.ones(100),
         paras = np.ones(17)
         ):
    """
    数据拟合所用的函数: A*cos(2*pi*k*x + theta)
    """

    return paras[1-1]*np.exp(-x/paras[6-1])*np.sin(2*np.pi*paras[10-1]*x+paras[14-1]) + \
           paras[2-1] * np.exp(-x / paras[7-1]) * np.sin(2 * np.pi * paras[11-1] * x + paras[15-1]) + \
           paras[3-1] * np.exp(-x / paras[8-1]) * np.sin(2 * np.pi * paras[12-1] * x + paras[16-1]) + \
           paras[4-1] * np.exp(-x / paras[9-1]) * np.sin(2 * np.pi * paras[13-1] * x + paras[17-1]) +\
           paras[5-1]

def residuals(p, y, x):
    """
    实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
    """
    return y - func(x, p)
data_path = 'C:\E\\USTC\MagLab\data\data\\'

filepath = data_path+'7742.txt'
data = np.loadtxt(filepath);
xdata = np.transpose(data[:2000,0])
ydata = np.transpose(data[:2000,1])

#A, k, theta = 10, 3, 6 # 真实数据的函数参数
#y0 = func(xdata, [A, k, theta]) # 真实数据
#y1 = y0 + 2 * np.random.randn(len(x)) # 加入噪声之后的实验数据

p0 = [1.466,0.46237,1.25863,0.487149,0.005,16.1833,17.95694763,15.1368830,15.34957978,\
     15.146803,4.443511,4.49156673,4.538369508,0,0,0,0.0] # 第一次猜测的函数拟合参数

# 调用leastsq进行数据拟合
# residuals为计算误差的函数
# p0为拟合参数的初始值
# args为需要拟合的实验数据
plsq = leastsq(residuals, p0, args=(ydata, xdata))

#print (u"真实参数:", [A, k, theta] )
print(plsq[0])  # 实验数据拟合后的参数

plt.figure()
plt.scatter(xdata, ydata,marker='x', color='b')
plt.plot(xdata, func(xdata, plsq[0]), color='g')

plt.show()