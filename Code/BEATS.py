# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:33:57 2016

GENERAL OVERVIEW 
------------------------------------------------------------------------------
Relevance overview
------------------------------------------------------------------------------
- One task of visual system to create normalized image
- This involves global search for min/max of inputs 
- Inputs are rescaled based on these values
- This exact method of normalization is not nuero-phys. plausible
- However, it could serve as inspiration for diffusion-like filling-in process
  to make the FACADE-like model by Gregory Francis more neuro-phys. plausible, 
  as well as, producing a more accurate output percept.

------------------------------------------------------------------------------
Meta-description of model
------------------------------------------------------------------------------
- Here exact normalization achieved by next-neighbour interactions
- Uses novel non-linear diffusuion systems (like filling-in) this is where
  Bigger EAts Smaller (BEATS), and regions of maximum brightness/darkness grow
  until a global min/max is reached.
- Reaction-diffusion system, Laplacian diffusion model a continuum of electrically
  couple cells (syncytium)
- Laplacian describes an electrical synapse (gap-junction) permitting
  hyperpolarizing and depolarizing somatic curents.
  
  Based on a smodified reaction-diffusion system (Laplacian diffusion)
  
------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Majority of the code is dedicated to solving various ordinary differential 
  equations (ODE's), which model various synaptic interactions.
- The solving method chosen here is the Fourth-Order Runga-Kutta (RK4) method. 
  One of the most robust and accurate method for solving ODE's. Keil tends to 
  use the Euler method to solve it, as it's simpler but less accurate. 
"""

import numpy as np
import sympy as sp
import scipy.ndimage.filters as flt
#import square_wave
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
    
def T(lamda,x): 
    """
    T Operator function
    ---------------------------------------------------------------------------
    lambda is a "steering" constant between 3 diffusion behaviour states, as a
    part of the diffusion operator. The consequences of steering being:
    
    -------------------------------------------------------
    |lamba |   state   | result on input x                |
    |------|-----------|----------------------------------|
    |==0   | linearity | Remains the same                 |
    |<0    | max       | Half-wave rectification          |
    |>0    | min       | Inverse half-wave rectification  |
    -------------------------------------------------------
    
    Parameters
    ------------    
    lambda : int    steering constant
    x :array-like   input stimulus arrau    
    
    """    
    if lamda == 0:  
        return x
    elif lamda > 0:
        maxval = np.zeros_like(x)
        return np.array([x, maxval]).max(axis=0)
    elif lamda < 0: 
        minval = np.zeros_like(x)
        return np.array([x, minval]).min(axis=0)
        

def Diffusion_operator(lamda,f,t):  
    """
    Diffusion Operator - 2D Spatially Discrete Non-Linear Diffusion
    ---------------------------------------------------------------------------
    This describes the synaptic states of diffusion processes and correspond to 
    It "models different types of electrical synapses (gap junctions).
    The linear version, Kλ=0 , describes the exchange of both polarizing and 
    de-polarizing". Thus the rectification found in the T operator serves to 
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
    |==0   | linearity  (T[0])       | Operator becomes Laplacian       |
    |<0    | positive K(lamda)       | Half-wave rectification          |
    |>0    | negative K(lamda)       | Inverse half-wave rectification  |
    ---------------------------------------------------------------------
    
    """
    
    if lamda == 0:  # linearity
        return flt.laplace(f)
    else:           # non-linearity (neighbour interactions up/down/left/right)
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

def Solve_diff_eq_RK4(stimulus,lamda,t0,h,N,D,t_N):
    """
    Finds a solution to the linear and spatially discrete diffusion equation:
    
                        df/dt = D*K(f) - s*delta(t-t0)  
                        
    using the 4th order Runge-Kutta method (ignoring leakage currents). The 
    stimulus is inject at t0 and then the defined lamda value permits certain
    diffusion behaviour to occur iteratively over this initial stimuli. The
    result is a diffusion of the maximum and minimum values across the stimuli
    to obtain a global value of luminance extremities through local interactions.   
    
    Parameters
    ---------------
    stimulus : array_like   input stimuli [x,y] "photo receptor activity elicited by
                            some real-world luminance distribution"
    lamda : int             a value between +/- inf
    t0 : int                point of stimulus "injection"
    h : int                 Runga-kutta step size
    N : int                 stimulus array size
    D : int                 diffusion coefficient weights rate of diffusion [0<D<1]
    
    Returns
    ----------------
    f : array_like          computed diffused array
    
    """
    f = np.zeros((t_N+1,N,N)) #[time, equal shape space dimension]
    t = np.zeros(t_N+1)
    
    if lamda ==0:
        """    Linearity  Global Activity Preserved   """
        for n in np.arange(0,t_N,h):
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


def ONtype_norm(s,t0,h,N,D,t_N,a,b,R,dt=1):
    """
    Dynamic normalisation or lightness filling-in
    ---------------------------------------------------------------------------
    Using the previously obtained global min values of the input stimuli
    a lightness diffusion process is used to obtain a normalized image of the 
    image.
    
    ISSUE : R (+1) regularisation parameter in steady state solution required
            to buffer NaN
    
    Returns
    --------
    c     - steady state solution
    c_out - dynamic solution
    """
    
    c = np.zeros((t_N,N,N))
    cd_out = np.zeros((t_N,N,N))

    for t in np.arange(0,t_N-1):
        c[t,:,:] =  (s-a[t,:,:])/(b[t,:,:]-a[t,:,:]+R)
    
        cd_out_1 = b[t,:,:]*(np.zeros((N,N))-cd_out[t,:,:])
        cd_out_2 = a[t,:,:]*(np.ones((N,N))-cd_out[t,:,:])
        cd_out[t,:,:] = dt*(cd_out_1 - cd_out_2 + stimulus)    
    
    return c , cd_out
    

def OFFtype_norm(t_N,N,a,b,c,s,R,dt=1):
    """
    Inverse dynamic normalisation or darkness filling-in
    --------------------------------------------------------------------------
    Same as dynamic normalisation, but opposite.
    
    ISSUE : R (+1) regularisation parameter in steady state solution to buffer NaN    
    
    Returns
    --------
    d     - steady state solution
    d_out - dynamic solution
    """
    
    d = np.zeros((t_N,N,N))
    d_out = np.zeros((t_N,N,N))
    for t in np.arange(0,t_N-1):
        d[t,:,:] = (b[t,:,:] - s) / (b[t,:,:] - a[t,:,:]+R)      
       
        d_out_1 = b[t,:,:]*(np.ones((N,N))-d_out[t,:,:])
        d_out_2 = a[t,:,:]*(np.zeros((N,N))-d_out[t,:,:])
        d_out[t,:,:] = dt*(d_out_1 - d_out_2 - s)
    return d, d_out
    

""" 
Here is the code to run
"""
# Parameters
D = 0.01  # Diffusion Coefficient
h = 1     # Runga-Kutta Step
t0 = 0    # Start time
t_N = 500 # End time = (Length of stimulation)
R = 1     # Regularisation parameter

# Import jpg image or use square wave stimulus
im = Image.open("/home/will/gitrepos/contAdaptTranslation/Documents/rs.png").convert('L')

# Resizing image (smaller) increases speed (but reduces accuracy)
arr = np.array(im.resize((400,400), Image.ANTIALIAS))

# Scale down from grey to binary scale
stimulus=arr/255.

#convert into usable array
N=arr.shape[0]

"""
To increase code speed, multiple processors are run simultaneously with different
diffusion states. The results are saved in the "results" array.
"""
def multi_run_wrapper(args):
   return Solve_diff_eq_RK4(*args)

# Three diffusion behaviour states
pool = Pool(4) # Open 4 processors available on this computer into Pools
state1=(stimulus,-1,t0,h,N,D,t_N)
state2=(stimulus,0,t0,h,N,D,t_N)
state3=(stimulus,1,t0,h,N,D,t_N)
results = pool.map(multi_run_wrapper,[state1,state2,state3])
pool.close()

# output results into arrays
a=results[0]  # minimum growth state of diffusion
b=results[2]  # maximum growth state of diffusion
ss=results[1] # steady-state diffusion

# Two diffusion layers converging to normalized image 
c, c_out = ONtype_norm(stimulus,t0,h,N,D,t_N,a,b,1) # Lightness filling-in
d, d_out = OFFtype_norm(t_N,N,a,b,c,stimulus,1)     # Darkness filling-in

maxval = np.zeros_like(c)


# Steady-state half-wave-rectify
S_bright = np.array([c, maxval]).max(axis=0)
S_dark   = np.array([d, maxval]).min(axis=0)

# Dynamic state half-wave-rectify
S_bright_d = np.array([c_out, maxval]).max(axis=0)
S_dark_d   = np.array([d_out, maxval]).min(axis=0)

# Perceptual activities
P = (S_bright-S_dark)/(1+S_bright+S_dark) # Steady-state
P_d = (S_bright_d-S_dark_d)/(1+S_bright_d+S_dark_d) # Dynamic

# Positive values only
#P  = np.array([P, maxval]).max(axis=0)
#P_d= np.array([P_d, maxval]).max(axis=0)



""" What is a percivable lightness increment? """








""" Plotting of outputs """

# Diffusion state plotter
plotter1=ss
plotter2=S_bright
plotter3=S_bright_d

plot_r=np.arange(1,t_N,50)
plot_max=0.1

f, axarr = plt.subplots(3, 6)
axarr[0, 0].imshow(plotter1[plot_r[0],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[0, 1].imshow(plotter1[plot_r[1],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[0, 2].imshow(plotter1[plot_r[2],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[0, 3].imshow(plotter1[plot_r[3],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[0, 4].imshow(plotter1[plot_r[4],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[0, 5].imshow(plotter1[plot_r[5],:,:], cmap='gray')#,vmax=1,vmin=0)

axarr[1, 0].imshow(plotter2[plot_r[0],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[1, 1].imshow(plotter2[plot_r[1],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[1, 2].imshow(plotter2[plot_r[2],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[1, 3].imshow(plotter2[plot_r[3],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[1, 4].imshow(plotter2[plot_r[4],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[1, 5].imshow(plotter2[plot_r[5],:,:], cmap='gray')#,vmax=1,vmin=0)

axarr[2, 0].imshow(plotter3[plot_r[0],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[2, 1].imshow(plotter3[plot_r[1],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[2, 2].imshow(plotter3[plot_r[2],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[2, 3].imshow(plotter3[plot_r[3],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[2, 4].imshow(plotter3[plot_r[4],:,:], cmap='gray')#,vmax=1,vmin=0)
axarr[2, 5].imshow(plotter3[plot_r[5],:,:], cmap='gray')#,vmax=1,vmin=0)

# Luminance edge profiler
plt.figure(2)#,figsize=[4,13])
plt.subplot(1,4,1)
plt.imshow(stimulus,cmap='gray')
plt.title('Input')
first_line=P[450,8,:]
second_line=P[450,13,:]
plt.subplot(1,4,2)
plt.plot(first_line,'r')
plt.plot(second_line,'b')
plt.title('Steady-state solution')
plt.subplot(1,4,3)
first_line=P_d[450,8,:]
second_line=P_d[450,13,:]
plt.plot(first_line,'r')
plt.plot(second_line,'b')
plt.title('Dynamic solution')
plt.subplot(1,4,4)
plt.imshow(c[450,:,:],cmap='gray')
plt.plot(np.arange(0,N,1),np.ones(N)*8,'r')
plt.plot(np.arange(0,N,1),np.ones(N)*13,'b')
plt.xlim([0,N])
plt.ylim([0,N])
plt.title('Output Percept')


#imag = d     # Image array to convert into video file
#imag_name = 'd_0.05' # Name of image to be saved to
#fps = 60            # Frames per second [3-5 default]
#
#imag_int8=(imag*255.).astype('uint8')  # Rescale back to RGB255 and change to uint8 format for avi 
#filename = "{0}{1}{2}".format('/home/will/Documents/Git_Repository/Outputs/',imag_name,'.avi')
#writer = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('M','J','P','G'), fps, (N, N), False)
#for i in np.arange(0,t_N):
#    writer.write(imag_int8[i,:,:])


## How the images evolve

## How the pixel distribution evolves
##ylim_max=2500
##xlim_max=1
##fig3, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.pyplot.subplots(ncols=6, figsize=(20,5))
##ax1.hist(np.reshape(f_out[1,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax1.set_xlim(0.0,xlim_max)
##ax1.set_ylim(0,ylim_max)
##ax2.hist(np.reshape(f_out[5,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax2.set_xlim(0.0,xlim_max)
##ax2.set_ylim(0,ylim_max)
##ax3.hist(np.reshape(f_out[10,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax3.set_ylim(0,ylim_max)
##ax3.set_xlim(0.0,xlim_max)
##ax4.hist(np.reshape(f_out[20,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax4.set_ylim(0,ylim_max)
##ax4.set_xlim(0.0,xlim_max)
##ax5.hist(np.reshape(f_out[50,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax5.set_ylim(0,ylim_max)
##ax5.set_xlim(0.0,xlim_max)
##ax6.hist(np.reshape(f_out[100,:,:],f_out.shape[1]*f_out.shape[2]),bins=20)
##ax6.set_ylim(0,ylim_max)
##ax6.set_xlim(0.0,xlim_max)


#N = 128
#stimulus = square_wave.square_wave((1,1), N, 1, 6, mean_lum=.5, period='ignore',start='high')


#f_out = Runge_Kutta(stimulus,0,t0,h,N,D,t_N) # Equilibrium
#a = Runge_Kutta(stimulus,-1,t0,h,N,D,t_N) # Minimum syncytiun (evolves into global minimum)
#b = Runge_Kutta(stimulus,1,t0,h,N,D,t_N) # Max syncytium (evolves into global maximum)

#
#im = Image.open("/home/will/gitrepos/contAdaptTranslation/Documents/whites.jpg").convert('L')
#arr = np.array(im.resize((500,500), Image.ANTIALIAS))
#stimulus=arr/255.
#
#plt.subplot(4,1,1)
#plt.imshow(stimulus,cmap='jet')
#plt.subplot(4,1,2)
#plt.imshow(stimulus,cmap='PiYG')
#plt.subplot(4,1,3)
#plt.imshow(stimulus,cmap='bwr')
#plt.subplot(4,1,4)
#plt.imshow(stimulus,cmap='Greens')





















