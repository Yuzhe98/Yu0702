import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

philim=2*np.pi
thetalim=np.pi

def sph2orth1d(vec=None):
    orth=np.zeros(3)
    orth[0]=vec[0]*np.sin(vec[1])*np.cos(vec[2])
    orth[1]=vec[0]*np.sin(vec[1])*np.sin(vec[2])
    orth[2]=vec[0]*np.cos(vec[1])
    return orth

def sph2orth(
        vec=None
):
    if len(vec.shape)==1:
        return sph2orth1d(vec)
    elif len(vec.shape)==2:
        xyz = np.zeros(np.shape(vec))
        xyz[:, 0] = vec[:, 0] * np.sin(vec[:, 1]) * np.cos(vec[:, 2])
        xyz[:, 1] = vec[:, 0] * np.sin(vec[:, 1]) * np.sin(vec[:, 2])
        xyz[:, 2] = vec[:, 0] * np.cos(vec[:, 1])
        return xyz
    else:
        raise ValueError('len(vec.shape)<=0 or len(vec.shape)>=3')

def Rad2Deg(
        vec=None
):
    r=vec
    r[:,1:]*=180.0/np.pi
    return r
def Rad2Deg1d(
        vec=None
):
    r=vec
    r[1:]*=180.0/np.pi
    return r
'''
def orth2sph(vec):
    x=vec[0]
    y=vec[1]
    z=vec[2]
    r=(x**2+y**2+z**2)**0.5
    if r==0:
        return [0,0,0]
    theta = np.arccos(z/r)
    (x ** 2 + y ** 2) ** 0.5
    if x>0 and y>0:
        phi = np.arcsin(y/)
        return [r, theta, phi]
    
    if x==0:
        phi = np.arcsin(y/(x**2+y**2)**0.5)
        return [r, theta, phi]
    else:
        phi = np.arcsin(y / (x ** 2 + y ** 2) ** 0.5)
        return [r, theta, phi]

    if y == 0:
        if x > 0:
            return [r, theta, 0]
        else:
            return [r, theta, np.pi]
'''
def AmpInter(
        loc=np.ones((6,3)),
        pert=np.ones(3),
        opt_abs=True
):
    loc_ortho=sph2orth(loc)
    pert_ortho=sph2orth1d(pert)
    amp=np.sum(\
        loc_ortho*pert_ortho,\
        axis=1)\
        /(loc[:,0]*pert[0])
    amp/=np.max(abs(amp))
    if opt_abs:return abs(amp)
    else:return amp

def DistInter(
        loc=None,
        pert=None
):
    loc_ortho = sph2orth(loc)
    dist = np.sum(
        loc_ortho * sph2orth1d(pert),axis=1)/pert[0]
    return dist-dist.min()

def TimeInter(
        loc=np.ones((6,3)),
        pert=np.ones(3),
        v=3e8
):
    dist=DistInter(loc=loc,pert=pert)
    return dist/v

def DircTriUniSamp(#opt_distri='TriUniform',
    theta_num=6,
    verbose_time=True,
    verbose_prog=False
):
    #check if phinum.type==int
    #to be written
    theta_seglen= thetalim / 2.0 / theta_num
    tri_edglen_half=2.0/np.sqrt(3)*np.sin(theta_seglen/2.0)
    samprec=np.zeros((1 + 2 * theta_num, 3), dtype=float)#theta,phi_start,phi_num
    samprec[:,0]=np.linspace(start=0, stop=thetalim, num=int(1 + 2 * theta_num), endpoint=True, dtype=float)
    samprec[0,1]=0
    samprec[0, 2] = 1
    samprec[2 * theta_num, 1] = 0
    samprec[2 * theta_num, 2] = 1
    for i in np.linspace(start=1, stop=2 * theta_num - 1, num=2 * theta_num - 1, endpoint=True, dtype=int):
        #if verbose_prog:print(samprec[i,0])
        phi_seglen=2.0*np.arcsin(tri_edglen_half/np.sin(samprec[i,0]))
        phi_segnum=int(philim / phi_seglen + 0.5)
        phi_seglen= philim / phi_segnum
        # theta,phi_start,phi_num
        samprec[i:, 1]=(i%2)*phi_seglen*0.5#phi_start
        samprec[i:, 2] = phi_segnum#phi_num
    return samprec

def DircRef(
    locations = np.ones((6,3),dtype=float),#radius,theta,phi
    theta_num=6,
    store_opt=False,
    store_path='C:\\',
    store_fname='',
    plot_opt=True,
    plot_scatter_opt=True,
    plot_vector_opt=True,
    verbose_time=True,
    verbose_prog=False
):
    loc_num=len(locations)
    N1 = 2 * theta_num + 1
    r = DircTriUniSamp(theta_num=theta_num)##theta,phi_start,phi_num
    direct_num=int(np.sum(r[:,2],axis=None))
    rec = np.zeros((direct_num, 2 + 2 * loc_num),dtype=float)#in a row: latitude, longtitude, (time, amp)*n
    ii=0
    for i1 in range(N1):
        theta=r[i1,0]
        #phi_start=r[i1,1]
        #phi_num=int(r[i1,2])
        phi_array=np.linspace(start=r[i1,1], stop=r[i1,1] + philim, num=int(r[i1, 2]), endpoint=False)
        for phii in range(len(phi_array)):
            phi=phi_array[phii]
            if verbose_time:
                tic = time.time()
            if verbose_prog:
                #print('velocity=', velocity[i1])
                print('theta=', theta)
                print('phi=', phi)
            p = np.array([1, theta, phi])
            t = TimeInter(loc=locations, pert=p, v=1.0)
            t /= t.max()
            a = AmpInter(loc=locations, pert=p)
            #a /= a.max()
            if verbose_prog:
                print('t=', t)
                print('a=', a)

            rec[ii,0] = theta
            rec[ii, 1] = phi
            rec[ii,2::2] =t
            rec[ii, 3::2]=a
            '''
                = np.array([theta, phi,
                            t[0], a[0], \
                            t[1], a[1], \
                            t[2], a[2], \
                            t[3], a[3], \
                            t[4], a[4], \
                            t[5], a[5] \
                            ], dtype=float).flatten()
            '''
            ii+=1
            if verbose_prog: print('theta, phi,(time, amp)*n=', rec[ii])
            if verbose_time:
                toc = time.time()
                N_remain = direct_num - ii
                t_remain = (toc - tic) * N_remain
                hour = int(t_remain // 3600)
                min = int((t_remain - 3600 * hour) // 60)
                second = t_remain - 3600 * hour - 60 * min
                print("remaining time=", hour, ':', min, ':', second, '\n\n')
    if store_opt:
        np.savetxt(store_path + store_fname + '.txt',rec)
    if plot_opt:
        #print('direct_num=',direct_num)
        directions=np.ones((direct_num,3),dtype=float)
        directions[:,1:]=rec[:,0:2]
        xyz=sph2orth(directions)
        if plot_vector_opt:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # Make the grid
            x, y, z = np.zeros(direct_num),np.zeros(direct_num),np.zeros(direct_num)
            # Make the direction data for the arrows
            u = xyz[:, 0]
            v = xyz[:, 1]
            w = xyz[:, 2]
            ax.quiver(x, y, z, u, v, w, length=1, arrow_length_ratio=0.1,normalize=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_zlim([-1.3, 1.3])
            plt.show()
        if plot_scatter_opt:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o')  #
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_zlim([-1.3, 1.3])
            plt.show()
    return rec,1*np.sin(np.pi/2/theta_num)*2

def DistCalcu(
        loc1_sph=np.zeros(3, dtype=float),
        loc2_sph=np.zeros(3, dtype=float)
):
    loc1_orth=sph2orth(loc1_sph)
    loc2_orth = sph2orth(loc2_sph)
    return np.sum((loc1_orth-loc2_orth)**2)**0.5

def AreaCalcu(
        loc_sph=np.zeros((3,3), dtype=float)
):
    loc_orth=sph2orth(loc_sph)
    v20=loc_orth[0]-loc_orth[2]
    v21 = loc_orth[1] - loc_orth[2]
    return np.sum(v20*v21)*0.5

def AreaCalcu3(
        loc0_sph=np.zeros(3, dtype=float),
        loc1_sph=np.zeros(3, dtype=float),
        loc2_sph=np.zeros(3, dtype=float)
):
    loc_sph=np.array([loc0_sph, loc1_sph, loc2_sph])
    return AreaCalcu(loc_sph=loc_sph)