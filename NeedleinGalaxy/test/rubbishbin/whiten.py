import numpy as np
import matplotlib.pyplot as plt
import time

def ExpMovMed(
        data=3.3*np.random.randn(40,97)+10,
        #samp_rate=512,
        seg_len = 1,
        seg_str = 0.3,
        pre_len = 100,
        Efactor = 0.4

):
    t_delta = seg_len - seg_str
    alpha = np.exp(-1.0 * Efactor * t_delta)
    pre_segnum = int(pre_len/t_delta)

    if pre_segnum<=0:
        raise ValueError('pre_segnum<=0')
    if (pre_segnum>=1000):
        print('pre_segnum>=1000')
        pre_segnum=1000
    #wei_arr = list(map(lambda x: alpha**x, np.arange(start=0, stop=pre_segnum, step=1)))
    #print('pre_segnum=',pre_segnum)
    y_len = len(data)
    #print('y_len=',y_len)
    x_len = len(data[0])
    #print('x_len=',x_len)
    wet_sum = (1 - np.power(alpha, pre_segnum)) / (1 - alpha)
    #print(wet_sum)
    EMA_matrix = np.ones((y_len, x_len), dtype=float)
    loop_times = pre_segnum
    #time_start = time.time()
    i=0
    ai = (alpha) ** (i) / wet_sum
    s=ai
    EMA_matrix[pre_segnum:] = ai * data[pre_segnum - i - 1:y_len - i - 1]
    for i in np.arange(start=1, stop=loop_times, step=1):
        ai = (alpha) ** (i)/wet_sum
        EMA_matrix[pre_segnum:] += ai * data[pre_segnum - i - 1:y_len - i - 1]
        s+=ai
    #time_end = time.time()

    #print('+ ', s)
    return EMA_matrix,pre_segnum

'''
def ExpMovAveMap(
        data=3.3 * np.random.randn(40, 97) + 10,
        samp_rate=512,
        seg_len=1,
        seg_str=0.3,
        pre_len=7,
        Elambda=0.5,
        # not often uesd
        pre_segnum=10,
        alpha=np.exp(-0.5)
):
    delta = seg_len - seg_str
    alpha = np.exp(-1.0 * Elambda * delta)
    pre_segnum = int(pre_len / delta)

    y_len = len(data)
    x_len = len(data[0])

    EMA_matrix = np.ones((y_len, x_len), dtype=float)
    loop_times = pre_segnum

    i = 0
    time_start = time.time()
    EMA_matrix[:, pre_segnum:] = data[:, pre_segnum - i - 1:x_len - i - 1]
    time_end = time.time()
    print('loop=', i, ' time cost=', time_end - time_start, 's')

    for i in range(loop_times - 1):
        time_start = time.time()
        ai = (alpha) ** (i + 1)

        EMA_matrix[:, pre_segnum:] += ai * data[:, pre_segnum - i - 2:x_len - i - 2]
        time_end = time.time()
        print('loop=', i + 1, ' time cost=', time_end - time_start, 's')

'''