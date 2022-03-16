import numpy as np
import math
import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import multiprocessing as mp
import pathos

def f(x,t):
    return -x**(3) + np.sin(t)
def dfdt(x,t):
    return 3*x**(5) - 3* x**(2)*np.sin(t) + np.cos(t)
h = 10**(-3) # grid spacing 
xa = 0
xb = 10
L = xb - xa
mx = int(L/h - 1)
x0 = 0 # intial condition

t_grid = np.arange(xa,xb+h,h)
x_array = np.zeros(len(t_grid))
for i,t in enumerate(t_grid):
    x_array[i] = x0
    k1 = h*f(x0,t)
    k2 = h*f(x0 + .5*k1,t+.5*h)
    k3 = h*f(x0+ .5*k2,t+.5*h)
    k4 = h*f(x0+k3,t+h)
    x0 += (k1 + 2*k2 + 2*k3 + k4)/6

plt.plot(t_grid,x_array)
plt.show()
