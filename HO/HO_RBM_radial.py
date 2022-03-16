import numpy as np
import math
import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import multiprocessing as mp
import pathos

def getPsi_r(n,l,nu):
    '''
    Definition of radial wavefunction taken from 
    https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#N-dimensional_isotropic_harmonic_oscillator
    
    Parameters
    ----------
    n : TYPE
        principle quantum number for radial equation
    l : TYPE
        angular momentum number l.
    nu : TYPE
        harmonic oscillator parameter from the potential term .5 \mu \omega^{2} r^{2}.

    Returns
    -------
    wf : TYPE
        radial wavefunction for the 3d-harmonic oscillator as a function of radius r. 

    '''
    lag = special.genlaguerre(n,l+.5)
    def wf(r):
        result = r**(l) * lag(2*nu*r) * (2 * nu * r**(2)) * np.exp(-nu*r**(2))
        return(result)
    return wf
def r(rho,m,omega):
    return rho/np.sqrt(m*omega)
def R(u,r):
    ## expects u to be an array (numerical sol to the ODE)
    return u/r
def energy(lamb, omega):
    return .5*omega*lamb
### Oscillator parameters (hbar = 1)
mu = 1.0
omega = 1.0
l_val = 0
n_val = 0
nu = mu*omega/2
### Get Wavefunction

psi = getPsi_r(n_val,l_val,nu)

### grid spacing
h = 10**(-4)

# define the domain
r_a = 0.01 # left boundary 
r_b = 5 # right boundary 

r_array = np.arange(r_a,r_b+h,h)
m = len(r_array)

#psi_array = psi(r_array)
#plt.plot(r_array,psi_array)
#plt.show()

## Now, we want to numerically solve the radial equation for the QHO
def F(r,coords,l,m,omega,E):
    R = coords[0]
    y = coords[1]
    Fx = y
    Fy = (-2*y/r + l*(l+1)/(r**2) + m**(2)*omega**(2)*r**2 - 2*m*E)*R
    result = np.array([Fx,Fy])
    return result


def solve_E(E,l,mass,omega,coords,grid_size):
    u = coords[0]
    v = coords[1]
    coords = np.array([u,v],dtype='float')
    l = float(l)
    sol = np.zeros((grid_size,2),dtype='float')
    for i,r in enumerate(r_array):
        sol[i] = coords
        k1 = h*F(r, coords,l, mass, omega, E)
        k2 = h*F(r +.5*h , coords+ .5*k1, l, mass, omega, E)
        k3 = h*F(r +.5*h, coords + .5*k2, l ,mass, omega, E)
        k4 = h*F(r + h, coords+k3, l, mass, omega, E)
        coords += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    return sol
init_coord = np.array([0,0])
psi_00 = solve_E(1.5*omega,l_val,mu,omega,init_coord,m)
plt.plot(r_array,psi_00[:,1])
plt.show()
'''
## main solver

E1 = 2
E2 = .1*omega 
u2 = solve_energy(E1, l_val)
target = 10**-2
while abs(E1-E2) > target:
    u1,u2 = u2,solve_energy(E2,l_val)
    E1,E2 = E2,E2 - u2*(E2-E1)/(u2-u1)

print(lamb1/(.1*omega))
'''
