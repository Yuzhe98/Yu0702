import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


elements = ["A","B","C","D","E"]  # 5个数

def exp_output(elements = ["A","B","C","D","E"]):
    if len(elements)==1:
        return elements
    if len(elements)==2:
        return ["("+elements[0]+elements[1]+")"]

    exp1 = exp_output(elements[1:])
    for i in range(len(exp1)):
        exp1[i] = "("+elements[0]+exp1[i]+")"

    exp0 = exp_output(elements[0:-1])
    for i in range(len(exp1)):
        exp0[i] = "("+exp0[i] + elements[-1]+")"
    return exp1+exp0

print(exp_output(elements = ["A","B","C","D","E"]))