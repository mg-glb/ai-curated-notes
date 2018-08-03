import numpy as np
from scipy import signal as sg

a = np.array([[1,2],[1,3]])
b = np.array([[1,1],[4,2]])

c = sg.convolve2d(a,b,mode='full',boundary='fill',fillvalue=-1)
print(c)