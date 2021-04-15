import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import interval

path='C:\E\Yu0702\\20190726\\'

earth_radius=6371e3
Berkeley=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(-122.2572)/180])
Fribourg=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(7.1575)/180])
Hayward=np.array([earth_radius,np.pi*(90-46.7930)/180,np.pi*(-122.0540)/180])
Krakow=np.array([earth_radius,np.pi*(90-37.6564)/180,np.pi*(19.9046)/180])
Mainz=np.array([earth_radius,np.pi*(90-50.0286)/180,np.pi*(8.2346)/180])
Hefei=np.array([earth_radius,np.pi*(90-31.8428)/180,np.pi*(117.2525)/180])

locations=np.array([Berkeley,Fribourg,Hayward,Krakow,Mainz,Hefei])

N1=1
N2=181
N3=360
N=N1*N2*N3
velocity=np.linspace(start=1,stop=1,num=N1,endpoint=True)
print(velocity)
latitude=np.linspace(start=0,stop=np.pi,num=N2,endpoint=True)
print(latitude)
longtitude=np.linspace(start=0,stop=2*np.pi,num=N3,endpoint=False)
print(longtitude)
rec=np.zeros((N,2+2*6))
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            tic=time.time()
            print('velocity=',velocity[i1])
            print('latitude=', latitude[i2])
            print('longtitude=', longtitude[i3])
            p=np.array([1,latitude[i2],longtitude[i3]])
            t=interval.TimeInter(loc=locations,pert=p,v=velocity[i1])
            t/=t.max()
            a=interval.AmpInter(loc=locations,pert=p)
            a /= a.max()
            print('t=',t)
            print('a=',a)
            rec[(N2 * N3) * i1 + N3 * i2 + i3]\
                =np.array([latitude[i2], longtitude[i3],
                t[0],a[0],\
                t[1],a[1],\
                t[2],a[2],\
                t[3],a[3],\
                t[4],a[4],\
                t[5],a[5]\
                        ],dtype=float).flatten()
            print(rec[(N2 * N3) * i1 + N3 * i2 + i3])
            toc=time.time()
            N_remain = N - ((N2 * N3) * i1 + N3 * i2 + i3 + 1)
            t_remain = (toc - tic) * N_remain
            hour = int(t_remain // 3600)
            min = int((t_remain - 3600 * hour) // 60)
            second = t_remain - 3600 * hour - 60 * min
            print("remaining time=", hour, ':', min, ':', second, '\n\n')
#np.savetxt(path + "20190726,181360ver02" + '.txt',rec)