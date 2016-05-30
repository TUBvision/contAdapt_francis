# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:33:57 2016

@author: will
"""
import numpy as np
import sympy as sp
import scipy.ndimage.filters as flt
import square_wave
import matplotlib as plt
from PIL import Image
# BEATS FILLING IN
"""
Task of visual system to create normalized image
- Global search for min/max of inputs simulating cellular adaptation
- Inputs are rescaled based on these values
- This exact method of normalization is not NPL plausible

- Here exact normalization achieved by next-neighbour interactions
- Uses novel non-linear diffusuion systems (like filling-in)
- Contrast borders are flooded with activity
- Reaction-diffusion system, Laplacian diffusion model a continuum of electrically
  couple cells (syncytium)
- Laplacian describes an electrical synapse (gap-junction) permitting
  hyperpolarizing and depolarizing somatic curents.
  
  
  
  Modified reaction-diffusion system (Laplacian diffusion)
"""


    
def T(lamda,x): 
    """
    T Operator
    lambda is a "steering" constant between 3 behaviour states
    -----------------------------
    0     -> linearity 
    +inf  -> max
    -inf  -> min
    -----------------------------
    """    
    if lamda == 0:  # linearity
        return x
    elif lamda > 0: #  Half-wave rectification
        return np.max(x,0)
    elif lamda < 0: # Inverse half-wave rectification
        return np.min(0,x)
        

def Diffusion_operator(lamda,f,t): # 2D Spatially Discrete Non-Linear Diffusion
    """
    Diffusion Operator
    ------------------------------------------
    Special case where lambda == 0, operator becomes Laplacian        
    
    
    Parameters
    ----------
    D : int                      diffusion coefficient
    h : int                      step size
    t0 : int                     stimulus injection point
    stimulus : array-like        luminance distribution     
    
    Returns
    ----------
    f : array-like               output of diffusion equation
    -----------------------------
    0     -> linearity (T[0])
    +inf  -> positive K(lamda)
    -inf  -> negative K(lamda)
    -----------------------------
    """
    if lamda == 0:  # linearity
        return flt.laplace(f)
    else:           # non-linearity
        f_new = np.zeros(f.shape)
        for x in np.arange(0,f.shape[0]-1):
            for y in np.arange(0,f.shape[1]-1):
                f_new[x,y]=T(lamda,f[x+1,y]-f[x,y]) + T(lamda,f[x-1,y]-f[x,y]) + T(lamda,f[x,y+1]-f[x,y])
                + T(lamda,f[x,y-1]-f[x,y])
        return f_new


def Dirac_delta_test(tester):
    # Limit injection to unitary multiplication, not infinite
    if np.sum(sp.DiracDelta(tester)) == 0:
        return 0
    else:
        return 1

def Runge_Kutta(stimulus,lamda,t0,h,N,D,t_N):
    """
    4th order Runge-Kutta solution to:
    linear and spatially discrete diffusion equation (ignoring leakage currents)
    
    Adiabatic boundary conditions prevent flux exchange over domain boundaries
        
    
    Parameters
    ---------------
    stimulus : array_like   input stimuli [t,x,y]
    lamda : int             0 +/- inf
    t0 : int                point of stimulus "injection"
    h : int                 step size
    N : int                 array size (stimulus.shape[1])
    D : int                 diffusion coefficient [constant]
    
    Returns
    ----------------
    f : array_like          computed diffused array
    
    """
    f = np.zeros((t_N+1,N,N)) #[time, equal shape space dimension]
    t = np.zeros(t_N+1)
    
    if lamda ==0:
        """    Linearity  Global Activity Preserved   """
        for n in np.arange(0,t_N):
            k1 = D*flt.laplace(f[t[n],:,:]) + stimulus*Dirac_delta_test(t[n]-t0)
            k1 = k1.astype(np.float64)
            k2 = D*flt.laplace(f[t[n]+(h/2.),:,:]+(0.5*h*k1)) + stimulus*Dirac_delta_test((t[n]+(0.5*h))- t0)
            k2 = k2.astype(np.float64)
            k3 = D*flt.laplace(f[t[n]+(h/2.),:,:]+(0.5*h*k2)) + stimulus*Dirac_delta_test((t[n]+(0.5*h))-t0)
            k3 = k3.astype(np.float64)
            k4 = D*flt.laplace(f[t[n]+h,:,:]+(h*k3)) + stimulus*Dirac_delta_test((t[n]+h)-t0)
            k4 = k4.astype(np.float64)
            f[n+1,:,:] = f[n,:,:] + (h/6.) * (k1 + 2.*k2 + 2.*k3 + k4)
            t[n+1] = t[n] + h
        return f
    
    else:
        """    Non-Linearity   (max/min syncytium) Global Activity Not Preserved   """
        for n in np.arange(0,t_N):
            k1 = D*Diffusion_operator(lamda,f[t[n],:,:],t[n]) + stimulus*Dirac_delta_test(t[n]-t0)
            k1 = k1.astype(np.float64)
            k2 = D*Diffusion_operator(lamda,(f[t[n]+(h/2.),:,:]+(0.5*h*k1)),t[n]) + stimulus*Dirac_delta_test((t[n]+(0.5*h))- t0)
            k2 = k2.astype(np.float64)
            k3 = D*Diffusion_operator(lamda,(f[t[n]+(h/2.),:,:]+(0.5*h*k2)),t[n]) + stimulus*Dirac_delta_test((t[n]+(0.5*h))-t0)
            k3 = k3.astype(np.float64)
            k4 = D*Diffusion_operator(lamda,(f[t[n]+h,:,:]+(h*k3)),t[n]) + stimulus*Dirac_delta_test((t[n]+h)-t0)
            k4 = k4.astype(np.float64)
            f[n+1,:,:] = f[n,:,:] + (h/6.) * (k1 + 2.*k2 + 2.*k3 + k4)
            t[n+1] = t[n] + h
            
        return f

#im = Image.open("/home/will/Downloads/lenaTest3.jpg")
#arr = np.array(im)
#arr=arr/253.

# Code to run
N=512   # Image size
stimulus = square_wave.square_wave((1,1), N, 1, 6, mean_lum=.5, period='ignore',start='high')
D = 0.05  #[0---1]
h = 1 #   h > 0
t0 = 0 # Injection time 
t_N = 500

f_out = Runge_Kutta(stimulus,0,t0,h,N,D,t_N)*6

f_out_first= f_out[0,:,:]
f_out_second= f_out[1,:,:]

# std of activity over time
f_out_std=np.std(f_out,2)
f_out_std=np.std(f_out_std,1)

# Average pixel activity
f_out_av=np.mean(f_out,2)
f_out_av=np.mean(f_out_av,1)

# How the images evolve
fig1, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.pyplot.subplots(ncols=7, figsize=(20,5))
ax1.imshow(f_out[1,:,:], cmap='gray')
ax2.imshow(f_out[10,:,:], cmap='gray')
ax3.imshow(f_out[25,:,:], cmap='gray')
ax4.imshow(f_out[50,:,:], cmap='gray')
ax5.imshow(f_out[100,:,:], cmap='gray')
ax6.imshow(f_out[250,:,:], cmap='gray')
ax7.imshow(f_out[500,:,:], cmap='gray')

# How the pixel distribution evolves
#ylim_max=2500
#xlim_max=1
#fig3, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.pyplot.subplots(ncols=6, figsize=(20,5))
#ax1.hist(np.reshape(f_out[1,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax1.set_xlim(0.0,xlim_max)
#ax1.set_ylim(0,ylim_max)
#ax2.hist(np.reshape(f_out[5,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax2.set_xlim(0.0,xlim_max)
#ax2.set_ylim(0,ylim_max)
#ax3.hist(np.reshape(f_out[10,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax3.set_ylim(0,ylim_max)
#ax3.set_xlim(0.0,xlim_max)
#ax4.hist(np.reshape(f_out[20,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax4.set_ylim(0,ylim_max)
#ax4.set_xlim(0.0,xlim_max)
#ax5.hist(np.reshape(f_out[50,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax5.set_ylim(0,ylim_max)
#ax5.set_xlim(0.0,xlim_max)
#ax6.hist(np.reshape(f_out[100,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
#ax6.set_ylim(0,ylim_max)
#ax6.set_xlim(0.0,xlim_max)

a = Runge_Kutta(stimulus,-1,t0,h,N,D,t_N)*6
b = Runge_Kutta(stimulus,1,t0,h,N,D,t_N)*6

def Global_normalization(stimulus,t0,h,N,D,t_N,a,b):
    """
    Dynamic normalization of image - reminicant of filling-in
    
    Steady-state solution to differential equation needs redefining
    """
    # steady state solution
    c = np.zeros((t_N+1,N,N))
    #cd_out = np.zeros((t_N+1,N,N))
    for t in np.arange(0,t_N):
        first = (stimulus-a[t,:,:])
        if np.sum(first) == 0: # avoid NaNs
            first = np.ones(first.shape)
        second =  (b[t,:,:]-a[t,:,:])
        c[t,:,:] = first/second  # normalized representation of stimulus
        #one = b[t,:,:]*(np.zeros(first.shape)-c[t,:,:])
        #two = a[t,:,:]*(np.ones(first.shape)-c[t,:,:])
        #cd_out[t,:,:] = one-two+stimulus    
    return c


c_out = Global_normalization(stimulus,t0,h,N,D,t_N,a,b)
#def Nonlinear_contrast
#    d = (b-s)/(b-a) # Off
#    
#    return [c,d]

plotter=c_out
fig3, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.pyplot.subplots(ncols=6, figsize=(20,5))
ax1.imshow(plotter[1,:,:], cmap='gray')
ax2.imshow(plotter[2,:,:], cmap='gray')
ax3.imshow(plotter[10,:,:], cmap='gray')
ax4.imshow(plotter[20,:,:], cmap='gray')
ax5.imshow(plotter[30,:,:], cmap='gray')
ax6.imshow(plotter[-2,:,:], cmap='gray')
