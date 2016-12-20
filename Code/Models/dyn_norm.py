"""
Author: Will Baker Morrison
Model: Matthias Keil

Simpler (Faster) version of BEATs.py using Euler method and ignoring extra
components. Input stimuli taken from "from tester_stim import SR_stimuli" with 
default input values. As it is written presently. SR_stim creates 7 SBC images
with stepped inner patch luminances as found in one of Mariannes papers. It then
processes these images seperately and plots the outputs

Critique
--------
-This code doesn't convert luminance [cd/m^2] into gray scale values
-Range of computed normalized ellipse values smaller than that of input values
-Only single point taken within the region [assumed test ptach uniformity]

References
---------
[1] Neural architectures for unifying brightness perception and image processing - Keil (2003)
[2] Local to global normalization dynamic by nonlinear local interactions - M.S. Keil
[3] Context affects lightness at the level of surfaces - M. Maertens 
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from PIL import Image
#from SR_stim import SR_stimuli
    
def T(lamda,x): 
    """
    T Operator function
    ---------------------------------------------------------------------------
    lambda is a "steering" constant between 2 diffusion behaviour states, as a
    part of the diffusion operator. The consequences of steering being:
    
    lamba <0 => max => Half-wave rectification          
    lamda >0 => min => Inverse half-wave rectification  

    Parameters
    ------------    
    lambda : int    steering constant
    x :array-like   input stimulus arrau    
    
    Returns
    ------------
    array : min/max of input value with zero
    
    """    
    
    if lamda > 0:
        maxval = np.zeros_like(x)
        return np.array([x, maxval]).max(axis=0)
    elif lamda < 0: 
        minval = np.zeros_like(x)
        return np.array([x, minval]).min(axis=0)
        

def Diffusion_operator(lamda,f):  
    """
    Diffusion Operator - 2D Spatially Discrete Non-Linear Diffusion
    ---------------------------------------------------------------------------
    The rectification found in the T operator serves to 
    dictate whether polizing (half-wave rectification) or de-polizing (inverse
    half-wave rectification) flow states are allowed to occur - assume
    state of surround, or enforce state on surround. 
    
    Parameters
    ----------
    lambda : int                 steering contstant
    f : array-like               input luminance distribution     
    
    Returns
    ----------
    f : array-like               output of diffusion equation    
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
D = 0.5   # Diffusion Coefficient [<0.75]
h = 1     # Runga-Kutta Step [h = 1]
t0 = 0    # Start time
t_N = 150 # Length of stimulation [up to 1000 or too much memory used]
R = 1     # Regularisation parameter [R = 1]
#stimulus=SR_stimuli() # Generate stimuli (see function for variations)

f_reduce=2# Resize images factor (for speed

output_lum_ellipse = np.zeros((7,4))
# Note: with 7 test stimuli this can take some time
for L in np.arange(7):
    # Shrink image for speed
    im = Image.fromarray(stimulus[L,:,:])
    arr = np.array(im.resize((im.size[0]/f_reduce,im.size[1]/f_reduce), Image.ANTIALIAS))
    
    # Scale down from grey to binary scale
    stim=arr/255. 
    
    """
    Solve diffusion equation with Euler method: a - darkness diffusion, b -
    lightness diffusion, tending towards global maximum and minimum
    """
    dt=1 # Euler method time step
    
    # a (Darkness diffusion layer)
    lamda = -1 # for T operator
    a = b =np.zeros_like(stim)
    for n in np.arange(0,t_N,dt):
        a = a + dt*(D*Diffusion_operator(lamda,a) + stim*Dirac_delta_test(n-t0))
    # b (Lightness diffusion layer)
    lamda = 1  # for T operator
    for n in np.arange(0,t_N):
        b = b + dt*(D*Diffusion_operator(lamda,b) + stim*Dirac_delta_test(n-t0))
    
    """ 
    Two diffusion layers recombined converging to normalized image: On-type
    normalization (Lightness filling-in), with steady-state and dynamic solutions
    """
    # Regularization parameter
    R=1 
    
    # Steady-state normalisation solution
    c =  (stim-a)/(b-a+R) 
    
    # Dynamic normalistion solution (solved with Euler method)
    cd_out = z = np.zeros_like(c)
    o = np.ones_like(b)
    cd_out = dt*(b*(z-cd_out) - a*(o-cd_out) + stim)    
    
    plt.figure('Final states as luminance profiles')
    plt.subplot(2,1,1)
    cross_y  = c.shape[0]/2
    plt.plot(c[cross_y,:])
    plt.title('Steady-state soln')
    plt.subplot(2,1,2)
    plt.plot(cd_out[cross_y,:])
    plt.title('Dynamic soln')
    
    # Change of ellipse and test patch luminance w.r.t. ellipse luminance.
    output_lum_ellipse[L,0]=c[c.shape[0]/2,c.shape[0]/2]
    output_lum_ellipse[L,1]=c[c.shape[0]/2,c.shape[0]*3/2]
    output_lum_ellipse[L,2]=cd_out[c.shape[0]/2,c.shape[0]/2]
    output_lum_ellipse[L,3]=cd_out[c.shape[0]/2,c.shape[0]*3/2]
    
    
# Plot example of change in Brightness in each diffusion layer
plt.figure('Diffusion layer change')
plt.subplot(2,1,1)
plt.imshow(a-stim,cmap='gray')
plt.title('Darkness change')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(b-stim,cmap='gray')
plt.title('Brightness change')
plt.colorbar()
    
# Plot change in luminance of ellipse
inpt = np.linspace(65,104,7)/255
plt.figure('Change in normalised inner patch luminance')
F=1
plt.subplot(2,1,1)
plt.plot(inpt,output_lum_ellipse[:,0]*F,'bx',label='left ss')
plt.plot(inpt,output_lum_ellipse[:,1]*F,'rx',label='right ss')
coeff = np.zeros((4,2))
for i in np.arange(2):
    coeff[i,:] = np.polyfit( inpt, output_lum_ellipse[:,i], 1 ) 
    p = np.poly1d( coeff[i,:] ) 
    x = np.linspace( 0, 0.45, 100 ) 
    plt.plot( x, p(x)) 
plt.legend()
plt.xlabel('Input Gray Values')
plt.ylabel('Normalized Gray Values')
plt.title('Steady state inner ellipse computation')

plt.subplot(2,1,2)
plt.plot(inpt,output_lum_ellipse[:,2]*F,'gx',label='left ds')
plt.plot(inpt,output_lum_ellipse[:,3]*F,'kx',label='right ds')
coeff = np.zeros((4,2))
for i in np.arange(2,4,1):
    coeff[i,:] = np.polyfit( inpt, output_lum_ellipse[:,i], 1 ) 
    p = np.poly1d( coeff[i,:] ) 
    x = np.linspace( 0, 0.45, 100 ) 
    plt.plot( x, p(x)) 
plt.legend()
plt.xlabel('Input Gray Values')
plt.ylabel('Normalized Gray Values')
plt.title('Dynamic inner ellipse computation')

