import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import interval
l=np.array([[2.3,0,0],[2.3,np.pi/3,0],[2.3,np.pi/3*2,0],[2.3,np.pi,0]])
p=np.array([2.3,np.pi,0])

interval.TimeInter(
        loc=l,
        pert=p,
        v=2.3
)
