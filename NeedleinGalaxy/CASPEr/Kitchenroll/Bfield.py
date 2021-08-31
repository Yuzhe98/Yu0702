import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

def arc(
        axis_x=0,
        axis_y=0,
        axis_z=1,
        R=(45 - 2.01/2)*10**(-3),
        phi0=0,
        phi1=120,
        #theta=120,
        lz=(238.96 + 1 - 59.04 - 1)*10**(-3),
        r=np.array([1,2,3]),
        winding=6,
        prec=5000,
        I=1,
        muo=4*np.pi*10**(-7)  # 1.25663706212*10**(-6)

):
    Bvec = np.zeros(3)
    phiarr = np.pi/180*np.linspace(start=phi0,stop=phi1,num=prec+1,endpoint=True)
    lxarr = R*np.cos(phiarr)
    lyarr = R*np.sin(phiarr)
    lzarr = lz*np.ones(prec+1)
    lvec = np.array([lxarr,lyarr,lzarr]).transpose()
    rpre = np.zeros(3)
    dlvec = np.zeros(3)
    #i=0
    #print(lvec[i])
    #print(lvec[i+1])
    for i in range(prec):
        dlvec=lvec[i+1]-lvec[i]
        rpre = r-(lvec[i+1]+lvec[i])/2
        Bvec+=np.cross(dlvec,rpre)/np.dot(rpre,rpre)**1.5
    return winding*muo*I/(4*np.pi)*Bvec  #
'''
circle=arc(
        R=1,
        phi0=0,
        phi1=360,
        lz=0,
        r=np.array([0,0,0]),
        prec=5000
)
print(circle)
'''
def rod(
        axis_x=0,
        axis_y=0,
        axis_z=1,
        R=(45 - 2.01/2)*10**(-3),
        theta=120,
        h0=0,
        h1=120,
        r=np.array([1,2,3]),
        prec=1000,
        I=1,
        muo=4*np.pi*10**(-7),  # 1.25663706212*10**(-6)
        winding=6
):
    Bvec = np.zeros(3)
    harr = np.linspace(start=h0,stop=h1,num=prec+1,endpoint=True)

    lxarr = R*np.cos(np.pi/180*theta)*np.ones(prec+1)
    lyarr = R*np.sin(np.pi/180*theta)*np.ones(prec+1)

    lvec = np.array([lxarr,lyarr,harr]).transpose()
    rpre = np.zeros(3)
    dlvec = np.zeros(3)
    for i in range(prec):
        dlvec=lvec[i+1]-lvec[i]
        rpre = r-(lvec[i+1]+lvec[i])/2
        Bvec+=np.cross(dlvec,rpre)/np.dot(rpre,rpre)**1.5
    return winding*muo*I/(4*np.pi)*Bvec  #
'''
infrod=rod(
        R=1,
        theta=0,
        h0=-1000,
        h1=1000,
        r=np.array([0,0,0]),
        prec=50000
)
print(infrod)
'''
def saddleL(
        axis_x=0,
        axis_y=0,
        axis_z=1,
        R=(45 - 2.01/2)*10**(-3),

        phileft0=30,
        phileft1=150,
        phiright0=210,
        phiright1=330,

        h0=-((238.96 + 1 - 59.04 - 1)/2)*10**(-3),
        h1=((238.96 + 1 - 59.04 - 1)/2)*10**(-3),
        prec=1000,
        prec_theta=1000,
        prec_h=1000,
        I=1,
        muo=4*np.pi*10**(-7),  # 1.25663706212*10**(-6)
        winding=6
):
    phileftarr = np.pi / 180 * (np.linspace(start=phileft0, stop=phileft1, num=prec_theta, endpoint=False)+0.5*(phileft1-phileft0)/prec_theta)
    nvec_left = np.array([-np.cos(phileftarr), -np.sin(phileftarr),  np.zeros(prec_theta)]).transpose()
    #print(nvec_left.shape)
    phirightarr = np.pi / 180 * (np.linspace(start=phiright0, stop=phiright1, num=prec_theta, endpoint=False) + 0.5 * (
                phiright1 - phiright0) / prec_theta)
    nvec_right = np.array([np.cos(phirightarr), np.sin(phirightarr), np.zeros(prec_theta)]).transpose()

    harr = np.linspace(start=h0, stop=h1, num=prec_h, endpoint=False)+0.5*(h1-h0)/prec_h
    area =(h1-h0)/prec_h * 2*np.pi*R*(phileft1-phileft0)/prec_theta/360
    PHI=0

    # left part
    for i in range(prec_theta):
        for j in range(prec_h):
            rh=np.array([-R*nvec_left[i,0],-R*nvec_left[i,1],harr[j]])
            rod1 = rod(R=R, theta=phileft0, h0=-g, h1=g, r=rh, prec=prec,winding=6)
            rod2 = rod(R=R, theta=phileft1, h0=g, h1=-g, r=rh, prec=prec,winding=6)
            rod3 = rod(R=R, theta=phiright0, h0=g, h1=-g, r=rh, prec=prec,winding=6)
            rod4 = rod(R=R, theta=phiright1, h0=-g, h1=g, r=rh, prec=prec,winding=6)

            arc1 = arc(R=R, phi0=phileft0, phi1=phileft1, lz=g, r=rh, prec=prec,winding=6)
            arc2 = arc(R=R, phi0=phiright1, phi1=phiright0, lz=g, r=rh, prec=prec,winding=6)
            arc3 = arc(R=R, phi0=phileft1, phi1=phileft0, lz=-g, r=rh, prec=prec,winding=6)
            arc4 = arc(R=R, phi0=phiright0, phi1=phiright1, lz=-g, r=rh, prec=prec,winding=6)
            Bvec = rod1+rod2+rod3+rod4+arc1+arc2+arc3+arc4
            PHI+=np.dot(Bvec,nvec_left[i])
    print("left L = ", PHI)
    '''
    # right part
    for i in range(prec_theta):
        for j in range(prec_h):
            rh=np.array([R*nvec_right[i,0],R*nvec_right[i,1],harr[j]])
            rod1 = rod(R=R, theta=phileft0, h0=-g, h1=g, r=rh, prec=prec)
            rod2 = rod(R=R, theta=phileft1, h0=g, h1=-g, r=rh, prec=prec)
            rod3 = rod(R=R, theta=phiright0, h0=g, h1=-g, r=rh, prec=prec)
            rod4 = rod(R=R, theta=phiright1, h0=-g, h1=g, r=rh, prec=prec)

            arc1 = arc(R=R, phi0=phileft0, phi1=phileft1, lz=g, r=rh, prec=prec)
            arc2 = arc(R=R, phi0=phiright1, phi1=phiright0, lz=g, r=rh, prec=prec)
            arc3 = arc(R=R, phi0=phileft1, phi1=phileft0, lz=-g, r=rh, prec=prec)
            arc4 = arc(R=R, phi0=phiright0, phi1=phiright1, lz=-g, r=rh, prec=prec)
            Bvec = rod1+rod2+rod3+rod4+arc1+arc2+arc3+arc4
            PHI+=np.dot(Bvec,nvec_right[i])
    
    print("left+right L = ", PHI)
    '''
    print("area = ", area)
    return 2*PHI*winding*area

g=((238.96 + 1 - 59.04 - 1)/2)*10**(-3)
radius =(45 - 2.01/2)*10**(-3)

sw=3

if sw==0:
    I = 1
    muo = 4 * np.pi * 10 ** (-7)  # 1.25663706212*10**(-6)
    winding = 6
    radius = (45 - 2.01 / 2) * 10 ** (-3)
    g = ((238.96 + 1 - 59.04 - 1) / 2) * 10 ** (-3)

    r0 = np.array([0.1, 0.1, 0.1])

    rod1 = rod(R=radius, theta=30, h0=-g, h1=g, r=r0, prec=5000)
    rod2 = rod(R=radius, theta=150, h0=g, h1=-g, r=r0, prec=5000)
    rod3 = rod(R=radius, theta=210, h0=g, h1=-g, r=r0, prec=5000)
    rod4 = rod(R=radius, theta=330, h0=-g, h1=g, r=r0, prec=5000)

    arc1 = arc(R=radius, phi0=30, phi1=150, lz=g, r=r0, prec=5000)
    arc2 = arc(R=radius, phi0=330, phi1=210, lz=g, r=r0, prec=5000)
    arc3 = arc(R=radius, phi0=150, phi1=30, lz=-g, r=r0, prec=5000)
    arc4 = arc(R=radius, phi0=210, phi1=330, lz=-g, r=r0, prec=5000)

    Bfield = winding * 10 ** (-7) * I * (rod1 + rod2 + rod3 + rod4 + arc1 + arc2 + arc3 + arc4)
    print(Bfield)

if sw==1:
    L=saddleL(
    axis_x=0,
    axis_y=0,
    axis_z=1,
    R=(45 - 2.01 / 2) * 10 ** (-3),

    phileft0=30,
    phileft1=150,
    phiright0=210,
    phiright1=330,

    h0=-g,
    h1=g,
    prec=100,
    prec_theta=100,
    prec_h=100,
    I=1,
    muo=4 * np.pi * 10 ** (-7),  # 1.25663706212*10**(-6)
    winding=6
)
    print("L = %g $\mu$H"%(10**6*L))

if sw==2:
    # Bz
    I = 1
    muo = 4 * np.pi * 10 ** (-7)  # 1.25663706212*10**(-6)
    winding = 6
    radius = (45 - 2.01 / 2) * 10 ** (-3)
    h_Bz = (170.50-125.50)/2 * 10 ** (-3)  # (187.97-110.03)/2 * 10 ** (-3)
    hzarr = np.array([h_Bz-0.01, h_Bz,h_Bz+0.01])

    zarr = np.linspace(start=-0.05,stop=0.05,num=10,endpoint=True)
    Bfield_hz=[]
    for hz in hzarr:
        Bfield=[]
        for z in zarr:
            arc0 = arc(R=radius, phi0=0, phi1=360, lz=hz, r=np.array([0.0, 0.0, z]), winding=9, prec=5000)
            arc1 = arc(R=radius, phi0=0, phi1=360, lz=-hz, r=np.array([0.0, 0.0, z]), winding=9, prec=5000)
            Bfield.append(arc0[2] + arc1[2])
        Bfield_hz.append(Bfield)
    labelarr = np.array(["h_Bz-0.01", "h_Bz", "h_Bz+0.01", "1 mM", "2 mM", "5 mM"])
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
    # fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    for i in range(3):
        ax0.plot(zarr,Bfield_hz[i], label=labelarr[i])  # ,c='grey'
        #
    ax0.set_xlabel('z-axis / m')  # , fontsize=14
    ax0.set_ylabel('Bz / T')  # , fontsize=14Absorption
    ax0.legend(loc='upper right')  # , fontsize=14
    plt.show()
    #print("Bfield = ",Bfield*10000," G")


if sw==3:
    # dBz/dz
    I = 1
    muo = 4 * np.pi * 10 ** (-7)  # 1.25663706212*10**(-6)
    winding = 6
    radius = (45 - 2.01 / 2) * 10 ** (-3)
    h_dBz = (187.97-110.03)/2 * 10 ** (-3)
    hzarr = np.array([h_dBz - 0.01, h_dBz, h_dBz + 0.01])

    zarr = np.linspace(start=-0.05, stop=0.05, num=10, endpoint=True)
    Bfield_hz = []
    for hz in hzarr:
        Bfield = []
        for z in zarr:
            arc0 = arc(R=radius, phi0=0, phi1=360, lz=hz, r=np.array([0.0, 0.0, z]), winding=9, prec=5000)
            arc1 = arc(R=radius, phi0=360, phi1=0, lz=-hz, r=np.array([0.0, 0.0, z]), winding=9, prec=5000)
            Bfield.append(arc0[2] + arc1[2])
        Bfield_hz.append(Bfield)
    labelarr = np.array(["h_Bz-0.01", "h_Bz", "h_Bz+0.01"])
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # , width_ratios=widths, height_ratios=heights
    # fig.subplots_adjust(left=0.193, top=0.95, right=0.967,bottom=0.18, wspace=0.198, hspace=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    for i in range(3):
        ax0.plot(zarr, Bfield_hz[i], label=labelarr[i])  # ,c='grey'
        #
    ax0.set_xlabel('z-axis / m')  # , fontsize=14
    ax0.set_ylabel('Bz / T')  # , fontsize=14Absorption
    ax0.legend(loc='upper right')  # , fontsize=14
    plt.show()