# -*- coding: utf-8 -*-
"""
Simpler (Faster) version of BEATs.py using Euler method and ignore extras

Input stimuli taken from "from tester_stim import SR_stimuli" with defaults

Critique
--------
This code doesn't convert luminance [cd/m^2] into gray scale values

Range of computed normalized ellipse values smaller than that of input values

Only single point taken within the region [assumed test ptach uniformity]

Code is relatively slow, especially if you create the stimuli on top
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
    
def T(lamda,x): 
    """
    T Operator function
    ---------------------------------------------------------------------------
    lambda is a "steering" constant between 3 diffusion behaviour states, as a
    part of the diffusion operator. The consequences of steering being:
    
    -------------------------------------------------------
    |lamba |   state   | result on input x                |
    |------|-----------|----------------------------------|
    |<0    | max       | Half-wave rectification          |
    |>0    | min       | Inverse half-wave rectification  |
    -------------------------------------------------------
    
    Parameters
    ------------    
    lambda : int    steering constant
    x :array-like   input stimulus arrau    
    
    """    
    
    if lamda > 0:
        maxval = np.zeros_like(x)
        return np.array([x, maxval]).max(axis=0)
    elif lamda < 0: 
        minval = np.zeros_like(x)
        return np.array([x, minval]).min(axis=0)
        

def Diffusion_operator(lamda,f,t):  
    """
    Diffusion Operator - 2D Spatially Discrete Non-Linear Diffusion
    ---------------------------------------------------------------------------
    The rectification found in the T operator serves to 
    dictate whether polizing (half-wave rectification) or de-polizing (inverse
    half-wave rectification) flow states are allowed to occur.
    
    Parameters
    ----------
    D : int                      diffusion coefficient
    h : int                      step size
    t0 : int                     stimulus injection point
    stimulus : array-like        luminance distribution     
    
    Returns
    ----------
    f : array-like               output of diffusion equation
    
    ---------------------------------------------------------------------
    |lamba |   state                 | result on input x                |
    |------|-------------------------|----------------------------------|
    |<0    | positive K(lamda)       | Half-wave rectification          |
    |>0    | negative K(lamda)       | Inverse half-wave rectification  |
    ---------------------------------------------------------------------
    
    """
    # non-linearity (neighbour interactions up/down/left/right)
    f_new = T(lamda,np.roll(f,1, axis=0)-f) \
    +T(lamda,np.roll(f,-1, axis=0)-f) \
    +T(lamda,np.roll(f, 1, axis=1)-f) \
    +T(lamda,np.roll(f,-1, axis=1)-f)
    return f_new


def Dirac_delta_test(tester):
    """
    The stimuli is injected at t=0 with a dirac delta function. This function
    limits the injection to a unitary multiplication, rather than infinite.
    """
    if np.sum(sp.DiracDelta(tester)) == 0:
        return 0
    else:
        return 1          
            
# Parameters
D = 0.5  # Diffusion Coefficient [<0.75]
h = 1     # Runga-Kutta Step [h = 1]
t0 = 0    # Start time
t_N = 200 # Length of stimulation [up to 1000 or too much memory used]
R = 1     # Regularisation parameter [R = 1]

output_lum_ellipse = np.zeros((7,4))
for L in np.arange(7):
    stim=stimulus[L,:,:]/255. # Scale down from grey to binary scale
    
    """
    Solve diffusion equation with Euler method: a - darkness diffusion, b -
    lightness diffusion, tending towards global maximum and minimum
    """
    dt=1 # Euler method time step
    
    # a (Darkness diffusion layer)
    lamda = -1
    a = b = np.zeros_like(stim)
    for n in np.arange(0,t_N,dt):
        a = a + dt*(D*Diffusion_operator(lamda,a,n) + stim*Dirac_delta_test(n-t0))
    # b (Lightness diffusion layer)
    lamda = 1    
    for n in np.arange(0,t_N):
        b = b + dt*(D*Diffusion_operator(lamda,b,n) + stim*Dirac_delta_test(n-t0))
    
    """ 
    Two diffusion layers recombined converging to normalized image: On-type
    normalization (Lightness filling-in), with steady-state and dynamic solutions
    """
    # Regularization parameter
    R=1 
    
    # Steady-state solution
    c =  (stim-a)/(b-a+R) 
    
    # Dynamic solution
    cd_out = z = np.zeros_like(c)
    o = np.ones_like(b)
    cd_out = dt*(b*(z-cd_out) - a*(o-cd_out) + stim)    
    
    # Plot Final States as Luminance Profiles
    plt.figure(1)
    plt.subplot(2,1,1)
    cross_y  = c.shape[0]/2
    plt.plot(c[cross_y,:])
    plt.title('Steady-state soln')
    plt.subplot(2,1,2)
    plt.plot(cd_out[cross_y,:])
    plt.title('Dynamic soln')
    
    # Change of ellipse and test patch luminance w.r.t. ellipse luminance.
    output_lum_ellipse[L,0]=c[150,150]
    output_lum_ellipse[L,1]=c[150,450]
    output_lum_ellipse[L,2]=cd_out[150,150]
    output_lum_ellipse[L,3]=cd_out[150,450]
    
    
# Plot example of change in Brightness in each diffusion layer
plt.figure(2)
plt.subplot(2,1,3)
plt.imshow(a-stim,cmap='gray')
plt.title('Darkness change')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(b-stim,cmap='gray')
plt.title('Brightness change')
plt.colorbar()
    
# Plot change in luminance of ellipse
inpt = np.linspace(65,104,7)/255
plt.figure(3)
F=1
plt.plot(inpt,output_lum_ellipse[:,0]*F,'bx',label='left ss')
plt.plot(inpt,output_lum_ellipse[:,1]*F,'rx',label='right ss')
plt.plot(inpt,output_lum_ellipse[:,2]*F,'gx',label='left ds')
plt.plot(inpt,output_lum_ellipse[:,3]*F,'kx',label='right ds')
plt.legend()
plt.xlabel('Input Gray Values')
plt.ylabel('Normalized Gray Values')
plt.title('Inner ellipse computation')
