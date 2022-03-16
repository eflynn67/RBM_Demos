import numpy as np
import math
import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import multiprocessing as mp
import pathos

def f(r,t):
    theta = r[0]
    omega =  r[1]
    ftheta = omega 
    fomega = -(g/l)*np.sin(theta)
    result = np.array([ftheta,fomega])
    return result

g = 9.81 
l = 0.1

h = 10**(-4) # grid spacing 
ta = 0
tb = 10


t_grid = np.arange(ta,tb+h,h)
mx = len(t_grid)
r = np.array([179.0,0.0]) ## initial 
sol = np.zeros((mx,2))

for i,t in enumerate(t_grid):
    sol[i] = r
    k1 = h*f(r,t)
    k2 = h*f(r + .5*k1,t+.5*h)
    k3 = h*f(r+ .5*k2,t+.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1 + 2*k2 + 2*k3 + k4)/6
    
plt.plot(t_grid,sol[:,0])
plt.show()