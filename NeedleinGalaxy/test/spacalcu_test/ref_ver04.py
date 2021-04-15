import numpy as np
import time
package_path ='C:\E\Yu0702\\NeedleinGalaxy'
import sys
sys.path.append(package_path)
import spacalcu

earth_radius=6371e3
Berkeley=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(-122.2572)/180])
Fribourg=np.array([earth_radius,np.pi*(90-37.8722)/180,np.pi*(7.1575)/180])
Hayward=np.array([earth_radius,np.pi*(90-46.7930)/180,np.pi*(-122.0540)/180])
Krakow=np.array([earth_radius,np.pi*(90-37.6564)/180,np.pi*(19.9046)/180])
Mainz=np.array([earth_radius,np.pi*(90-50.0286)/180,np.pi*(8.2346)/180])
Hefei=np.array([earth_radius,np.pi*(90-31.8428)/180,np.pi*(117.2525)/180])
locations=np.array([Berkeley,Fribourg,Hayward,Krakow,Mainz,Hefei])

spacalcu.DircRef(
    locations = locations,#radius,theta,phi
    theta_num=5,
    store_opt=False,
    store_path='C:\E\Yu0702\\NeedleinGalaxy\\test\\',
    store_fname='ref_ver04',
    plot_opt=True,
    plot_vector_opt=True,
    plot_scatter_opt=False,
    verbose_time=True,
    verbose_prog=False
)
'''
in ref ver04
theta_num=5
90/5=16
radius=1*np.sin(16/2/180*np.pi)*2
'''