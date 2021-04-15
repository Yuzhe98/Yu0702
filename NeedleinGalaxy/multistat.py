import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('C:\E\Yu0702\pYUackages')
import epa
import spacalcu
def locating(
    s_serieses=None,
    t_len=3600,
    samp_rate=512,
    seg_len=2,
    N_sens=6,
    lopass=0,
    hipass=5,
    confidc=None,
    prpr=False
):
    """
    Pick up signals according to the input threshold
    """
    rec = np.array([np.zeros(2*confidc[0])])
    for i in range(N_sens-1):
        #print(rec)
        rec = np.append(rec, [np.zeros(2*confidc[i+1])], axis=0)
    for sensi in range(N_sens):
        tic = time.time()
        s_whitened,aband\
                    =epa.ExPow(t_data=s_serieses[sensi],  # signal series in time domain
                                    samp_rate=samp_rate,  # sampling rate
                                    t_len=t_len,  #length of measuring time
                                    seg_len=seg_len,  # Length of each seg in seconds
                                    seg_strd=seg_len/2.0,  # overlap length, sgement stride in seconds
                                    f_lpass=lopass,
                                    f_hpass=hipass,
                                    avg_opt=True,
                                    avg_method='expmovmed',
                                    EMAwin=50*seg_len,  #100
                                    EMAfcor=0.02,  #0.1+0.1*j
                                    make_plot=False,  # Condition to produce plots
                                    colorbar_log=True,
                                    retopt=True)
        axis1sum = np.sum(s_whitened, axis=1)
        s_time=np.linspace(start=(aband+1)*seg_len/2,stop=t_len-seg_len/2,num=len(axis1sum),endpoint=True)
        rank=np.sort(axis1sum)
        if rank[-confidc[sensi]]==rank[-confidc[sensi]-1]:
            raise ValueError('rank[-confidc[sensi]]==rank[-confidc[sensi]-1]')
        threshold=rank[-confidc[sensi]]
        mark=0
        for j in range(len(axis1sum)):
            if axis1sum[j]>=threshold:
                rec[sensi][0+mark*2]=s_time[j]
                rec[sensi][1+mark*2]=axis1sum[j]
                mark+=1
            if mark>=confidc[sensi]: break
        #time record
        toc=time.time()
        N_remain=N_sens-sensi
        t_remain=(toc-tic)*N_remain
        hour=int(t_remain//3600)
        min=int((t_remain-3600*hour)//60)
        second=t_remain-3600*hour-60*min
        print("locating.py remaining time=", hour, ':', min, ':', second, '\n\n')
    print('rec=',rec)
    return rec
def lin01(
    arr=None
):
    arr-=min(arr)
    if max(arr!=0):
        arr/=max(arr)
    return arr
def MFind(
    data=None,
    max_num=0,
    min_num=0
):
    if max_num==0 and min_num==0:
        raise ValueError('max_num==0 and min_num==0')
    #data_num=len(data)
    rank=np.sort(data)

    if max_num == 0 and min_num >= 1:
        min_rec=np.zeros(min_num,dtype=int)
        min_thd = rank[min_num-1]#min_threshold
        if rank[min_num-1]==rank[min_num]:
            print('rank[min_num-1]==rank[min_num]')
        mini=0
        for i in range(len(data)):
            if data[i]<=min_thd:
                #print('min=',data[i])
                min_rec[mini]=i
                mini+=1
            if mini>=min_num:
                return min_rec
    if max_num >= 1 and min_num == 0:
        max_rec = np.zeros(max_num, dtype=int)
        max_thd = rank[-max_num]  # min_threshold
        if rank[-max_num] == rank[-max_num-1]:
            print('rank[-max_num]==rank[-max_num-1]')
        maxi = 0
        for i in range(len(data)):
            if data[i] >= max_thd:
                max_rec[maxi] = i
                maxi+=1
            if maxi >= max_num:
                return max_rec
    if max_num >= 1 and min_num >= 1:
        min_rec = np.zeros(min_num, dtype=int)
        max_rec = np.zeros(max_num, dtype=int)

        min_thd = rank[min_num - 1]# min_threshold
        max_thd = rank[-max_num]# max_threshold

        if rank[min_num-1]==rank[min_num]:
            print('rank[min_num-1]==rank[min_num]')
        if rank[-max_num] == rank[-max_num - 1]:
            print('rank[-max_num]==rank[-max_num-1]')

        mini = 0
        maxi = 0

        for i in range(len(data)):
            if data[i] <= min_thd:
                min_rec[mini] = i
                mini+=1
            elif data[i] >= max_thd:
                max_rec[maxi] = i
                maxi+=1
            if mini >= min_num and maxi>=max_num:
                return min_rec,max_rec
    raise ValueError('max_num and min_num went wrong')
    return -1
def QuickExam(
    ta,#time interval and amp
    ref=None,
    time_exm_opt=True,
    amp_exm_opt=False,
    exm_num=3,
    theta_num=5,
    ans=None,
    ans_opt=True,
    plot_opt=True,
    verbose_prog=False
):
    #timeinter=lin01(ta[0::2])
    amp = ta[1::2]
    #ref_timeinter=ref[:,2::2]
    #time loc
    if time_exm_opt:
        delta=ref[:,2::2]-lin01(ta[0::2])
        delta=np.sum(delta**2,axis=1)
        min_loc=MFind(delta,min_num=exm_num)
    if amp_exm_opt:
        print('sorry. amplitude examination is to be developed')
    # search result direction
    sr = np.ones((exm_num, 3), dtype=float)
    for i in range(exm_num):
        sr[i, 1:] = ref[min_loc[i],0:2]
    if verbose_prog:
        print('sr_rad=', sr)
        print('ans_rad=',ans)
        print('max(sr[:,1])>=ans[0]  ',max(sr[:,1])>=ans[0])
        #print('max(sr[:,1]), ans[0]=',max(sr[:,1]), ans[0])
        print('min(sr[:,1])<=ans[0]  ',min(sr[:,1])<=ans[0])
        #print('min(sr[:,1], ans[0]=', min(sr[:, 1]) ,ans[0])
        print('max(sr[:, 2]) >= ans[1]  ',max(sr[:, 2]) >= ans[1] )
        #print('max(sr[:, 2]), ans[1]=', max(sr[:, 2]), ans[1])
        print('min(sr[:, 2]) <= ans[1]  ', min(sr[:, 2]) <= ans[1])
        #print('min(sr[:, 2]), ans[1]=', min(sr[:, 2]), ans[1])

    # search result direction
    '''
    if max(sr[:,1])>=ans[0] and min(sr[:,1])<=ans[0] and\
        max(sr[:, 2]) >= ans[1] and min(sr[:, 2]) <= ans[1]:
        if verbose_prog:print('right')
        ret= 0
    else:
        if verbose_prog:print('wrong')
        plot_opt=True
        print('sr=', sr)
        print('ans=', ans)
        print('max(sr[:,1])>=ans[0]  ', max(sr[:, 1]) >= ans[0])
        # print('max(sr[:,1]), ans[0]=',max(sr[:,1]), ans[0])
        print('min(sr[:,1])<=ans[0]  ', min(sr[:, 1]) <= ans[0])
        # print('min(sr[:,1], ans[0]=', min(sr[:, 1]) ,ans[0])
        print('max(sr[:, 2]) >= ans[1]  ', max(sr[:, 2]) >= ans[1])
        # print('max(sr[:, 2]), ans[1]=', max(sr[:, 2]), ans[1])
        print('min(sr[:, 2]) <= ans[1]  ', min(sr[:, 2]) <= ans[1])
        # print('min(sr[:, 2]), ans[1]=', min(sr[:, 2]), ans[1])
        ret= 1
    '''
    d01=spacalcu.DistCalcu(
        loc1_sph=sr[0],
        loc2_sph=sr[1]
    )
    d02=spacalcu.DistCalcu(
        loc1_sph=sr[0],
        loc2_sph=sr[2]
    )
    d12=spacalcu.DistCalcu(
        loc1_sph=sr[2],
        loc2_sph=sr[1]
    )
    #std=(d01-r)**2+(d02-r)**2+(d12-r)**2#np.min((sr[:,1]-ans[0])**2)+np.min((sr[:,2]-ans[1])**2)
    #inter_max = 1 * np.sin(np.pi / 2 / theta_num) * 2
    #edge_len = np.sqrt(3) * np.sin(np.pi / 4.0 / theta_num)
    tri_area=2.0 * np.sin(np.pi / 4.0 / theta_num)*np.sqrt(3)

    std = spacalcu.AreaCalcu(loc_sph=sr)-tri_area
    print('tri_area=', tri_area)
    print('d01=',d01)
    print('d02=', d02)
    print('d12=',d12)
    print('spacalcu.AreaCalcu(loc_sph=sr)=', spacalcu.AreaCalcu(loc_sph=sr))
    print('std=',std)
    if min(d12,d02,d01)<1e-4:
        print('sr=',sr)
        plot_opt=True
    if std<=0.1*tri_area:
        if verbose_prog: print('right')
        ret = 0
    else:
        print(std)
        if verbose_prog: print('wrong')
        plot_opt = True
        print('sr=', sr)
        print('ans=', ans)
        ret = 1

    if plot_opt:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        direct_num = len(ref)
        ref_scats = np.ones((direct_num, 3), dtype=float)
        ref_scats[:, 1:] = ref[:, 0:2]
        ref_xyz = spacalcu.sph2orth(ref_scats)
        ax.scatter(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], marker='o',s=10)
        sr_orth = spacalcu.sph2orth(sr)
        if verbose_prog: print('sr_xyz=', sr_orth)
        ax.scatter(sr_orth[:,0],sr_orth[:,1],sr_orth[:,2],color='r',marker='P',s=300*np.exp(-delta))
        
        x, y, z = np.zeros(1), np.zeros(1), np.zeros(1)
        ans_sph = np.ones(3, dtype=float)
        ans_sph[1:] = ans
        ans_orth = spacalcu.sph2orth1d(ans_sph)
        #print('ans_orth=',ans_orth)
        u = np.array(ans_orth[0])
        v = np.array(ans_orth[1])
        w = np.array(ans_orth[2])
        ax.quiver(x, y, z, u, v, w, \
                  length=1, arrow_length_ratio=0.3, normalize=True,color='g',\
                  linewidth=4)
        #ax.scatter(ans_orth[0], ans_orth[1], ans_orth[2], color='g',marker='P',s=200)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        xyzlim=1.2
        ax.set_xlim([-xyzlim, xyzlim])
        ax.set_ylim([-xyzlim, xyzlim])
        ax.set_zlim([-xyzlim, xyzlim])
        plt.show()
    return ret,std

'''
def DircDens(
        ta,
        stat_num=6,
        sig_num=np.array([1, 1, 1, 1, 1, 1], dtype=int),
        ref_path='C:\E\Yu0702\\20190726\\',
        ref_fname='20190726,181360ver02',
        ans=None,
        ans_opt=True,
        plot_opt=True,
        colormap_name='plasma',
        colorbar_log=False,

        Lz_opt=True,
        plot_FWHM=0.001,
        ):
    ref = np.loadtxt(ref_path + ref_fname + '.txt')
    theta_array = ref[:, ]
    loopnum=1
    for i in range(stat_num):
        loopnum*=sig_num[i]
    densmap=np.zeros(N_la*N_lo)
    looprec =0
    for i0 in range(sig_num[0]):
        for i1 in range(sig_num[1]):
            for i2 in range(sig_num[2]):
                for i3 in range(sig_num[3]):
                    for i4 in range(sig_num[4]):
                        for i5 in range(sig_num[5]):
                            tic=time.time()
                            exm=np.array([ta[0][i0*2:i0*2+2],\
                                          ta[1][i1*2:i1*2+2],\
                                          ta[2][i2*2:i2*2+2],\
                                          ta[3][i3*2:i3*2+2],\
                                          ta[4][i4*2:i4*2+2],\
                                          ta[5][i5*2:i5*2+2]])
                            #print('exm=',exm)
                            exm[:, 0] -= np.min(exm[:, 0],axis=0)
                            exm[:, 0] /= np.max(exm[:, 0],axis=0)
                            exm[:, 1] /= np.max(exm[:, 1],axis=0)
                            exm=np.transpose(exm)
                            #print('processed rec=', exm)
                            densmap+=exam(exm, ref_path=ref_path, fname=ref_fname)
                            toc=time.time()
                            looprec+=1
                            N_remain = loopnum - looprec
                            t_remain = (toc - tic) * N_remain
                            hour = int(t_remain // 3600)
                            min = int((t_remain - 3600 * hour) // 60)
                            second = t_remain - 3600 * hour - 60 * min
                            print("remaining time=", hour, ':', min, ':', second, '\n\n')


    if plot_opt:
        fig = plt.figure()
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        densmap_ax = fig.add_subplot(gs[0, 0])
        densmap_forfig=np.reshape(densmap,(N_la,N_lo))
        if Lz_opt:
            densmap_forfig = 1.0/(1+(densmap_forfig/plot_FWHM*0.5)**2)
        densmap_forfig/=np.max(densmap_forfig)
        if colorbar_log:
            c = densmap_ax.pcolormesh(180/np.pi*ref[0:N_lo,1], 180/np.pi*ref[::N_lo,0],densmap_forfig , \
                                   norm=LogNorm(vmin=densmap_forfig.min(), vmax=densmap_forfig.max()), cmap=colormap_name)
        else:
            c = densmap_ax.pcolormesh(180/np.pi*ref[0:N_lo,1],180/np.pi*ref[::N_lo,0], densmap_forfig, \
                                   vmin=densmap_forfig.min(), vmax=densmap_forfig.max(), cmap=colormap_name)
        densmap_ax.set_ylim([182, -2])
        densmap_ax.set_xlim([-2,361])
        densmap_ax.set_title('density map', fontsize=20)
        densmap_ax.set_xlabel('Longtitude', fontsize=20)
        densmap_ax.set_ylabel('Latitude', fontsize=20)
        fig.colorbar(c, ax=densmap_ax)
        if ans_opt:
            for ansi in range(len(ans)):
                densmap_ax.axvline(180/np.pi*ref[ans[ansi]][1],alpha=0.8)
                densmap_ax.axhline(180/np.pi*ref[ans[ansi]][0],alpha=0.8)

        fig.tight_layout()
        plt.show()
    return 0
'''
def singlepeakexm():
    return 0