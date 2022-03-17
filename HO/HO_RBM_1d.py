import numpy as np
import math
import scipy as sci
from scipy import integrate
from scipy import special
import matplotlib.pyplot as plt
import multiprocessing as mp
import pathos
### NOTE: hbar = 1 in this demo
### First define exact solutions to compare numerical solutions to.
def getPsi_x(n,nu):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.
    
    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    nu : TYPE
        harmonic oscillator parameter sqrt( 1/mass*omega) from the potential term .5 \mu \omega^{2} r^{2}.

    Returns
    -------
    wf : TYPE
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x. 

    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*nu))*np.exp(-x**(2)/(2*nu**(2)))*herm(x/nu)
        return(result)
    return wf
def getExactE(n,omega):
    '''
    Exact energies of the 1d harmonic oscillator. 

    Parameters
    ----------
    n : float
        principle quantum number. float integers
    omega : float
        oscillator frequency.

    Returns
    -------
    float
        oscillator energy. 

    '''
    return (n+ .5)*omega
###  HO parameters
mu = 1.0
omega = 1.0
n_max = 3 # maximum number of HO states to consider
# define oscillator length scale x0 
x0 = np.sqrt(1/mu*omega)
# define n array
n_vals = np.arange(0,n_max,1)
x0_vals = np.linspace(x0, 5*x0,5)

### grid spacing
h = 10**(-3)

# define the domain
x_a = -5 # left boundary 
x_b = 5 # right boundary 

x_array = np.arange(x_a,x_b+h,h)
m = len(x_array) 
print(m)
## Now numericaly solve the 1-d SE. We do this in 2 ways. First we construct 
## the discretized representation of the Hamiltonian H and use LA to diagonalize it
## Second, we use an iterative scheme to solve the ODE without explictly constructing
## the matrix H. 
def V(x,x0):
    return .5*(x**2)/x0**4

off_diag = np.zeros(m)
off_diag[1] = 1

H = -1*(-2*np.identity(m) + sci.linalg.toeplitz(off_diag))/(2*mu*h**2) + np.diag(V(x_array,x0))

evals,evects = np.linalg.eigh(H)
norm = np.sqrt(1/sci.integrate.simpson(evects[:,0]*evects[:,0],x_array))
print(norm)
plt.plot(x_array,evects[:,0]*norm)
plt.show()
'''

#### Now numericaly solve the 1-d SE. We use 4th order Runge-Kutta here.
#### define 'inhomogenous' part of system of first order odes
def f(r,x,mass,energy,nu):
    psi = r[0]
    phi =  r[1]
    f_psi = phi
    f_phi = (-2*mass*energy + (x**2)/nu**4)*psi
    result = np.array([f_psi,f_phi])
    return result
## Calculate the wavefunction for particular energy E

def solve(E,r0,mass,nu,grid):
    mx = grid.shape[0]
    sol = np.zeros((mx,2))
    r = r0.copy()
    for i,x in enumerate(grid):
        sol[i] = r
        k1 = h*f(r,x,mass,E,nu)
        k2 = h*f(r + .5*k1,x+.5*h,mass,E,nu)
        k3 = h*f(r+ .5*k2,x+.5*h,mass,E,nu)
        k4 = h*f(r+k3,x+h,mass,E,nu)
        r += (k1 + 2*k2 + 2*k3 + k4)/6
    return sol,r[0]
init_coords = np.array([0.0,1.0])


### Note that the SE gives us an equation g(E) = psi(boundary value) at x = L 
### for a particular eigenvalue E. For our case, the boundary is at x = inf and psi = 0
### so we just need to find the root of this function.
## We use the secant method to find roots

## set boundary values for search domain
E1 = 0.0
E2 = .6

wf_sol_init,boundary2 = solve(E1,init_coords,mu,x0,x_array)
wf_exact = psi(x_array)


eps = 10**(-5)
while abs(E2 - E1) > eps:
    print(abs(E2-E1))
    boundary1, (wf_sol, boundary2) = boundary2, solve(E2,init_coords,mu,x0,x_array)
    E1,E2 = E2, E2 - boundary2*(E2-E1)/(boundary2 - boundary1 )
norm = np.sqrt(integrate.simpson(wf_sol[:,0]**2,x_array))
wf_sol = wf_sol/norm
exact_E = getExactE(n_val, omega)
print('Exact Energy: ',exact_E)
print('Numerical Energy: ', E2)


plt.plot(x_array,wf_sol[:,0],label='RK sol')
plt.plot(x_array,wf_exact,label='exact')
plt.legend()
plt.show()
'''
