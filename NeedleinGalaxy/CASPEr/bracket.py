import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec



orders = np.zeros((24,4),dtype=int)

for i in range(4):
    for j in range(3):
        for k in range(2):
            orders[i*4+j*3+k*2,0] = i
            orders[i * 4 + j * 3 + k * 2, 1] = j
            orders[i * 4 + j * 3 + k * 2, 2] = k
            orders[i * 4 + j * 3 + k * 2, 3] = 0


for j in range(24):
    elements = ["A", "B", "C", "D", "E"]  # 5个数
    operators = ["+", "-", "×", "÷"]  # 需要事先枚举出所有可能的运算符
    for i in orders[j]:
        elements[i] = "(" + elements[i] + operators[i] + elements[i + 1] + ")"
        elements.remove(elements[i + 1])
        operators.remove(operators[i])
    print(elements[0])


