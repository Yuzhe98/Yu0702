import numpy

def even(len = 2
        ):
    len=int(len)
    if len<=0: raise ValueError("incorrect input length of windowing")
    window = numpy.ones(len,dtype=float)
    return window

def hanning(len = 2,
            a0=0.5
            ):
    len=int(len)
    if len<=0: raise ValueError("incorrect input length of windowing")
    n=numpy.arange(len,dtype=float)
    a1_=a0-1
    window=numpy.zeros(len)
    window=a1_*numpy.cos(2*numpy.pi*n/float(len)) + a0
    return window
