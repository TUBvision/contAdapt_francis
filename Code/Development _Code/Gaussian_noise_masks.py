# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:24:04 2016

@author: will

Filtering an image with Gaussian Noise Masks
"""
import numpy as np
import math
import matplotlib.pylab as plt
from scipy.ndimage.filters import convolve

def Gaussian2D(GCenter, Gamp, Ggamma,Gconst): #new_theta > 0.4:
    """
    Produces a 2D Gaussian pulse    *EDITED BY WMBM
    
    Parameters
    ----------
    GCenter : int
            Centre point of Gaussian pulse
    Gamp : int
            Amplitude of Gaussian pulse
    Ggamma : int
            FWHM of Gaussian pulse
    Gconst : float
            Unkown parameter of density function
    
    Returns
    ----------
    GKernel : array_like
            Gaussian kernel
    """
    new_theta = math.sqrt(Gconst**-1)*Ggamma
    SizeHalf = np.int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    part1=(x-GCenter[0])**2+(y-GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)
    return GKernel
    
def conv2(x,y,mode='same'):
    """
    Emulate the Matlab function conv2 from Mathworks.
    
    Usage:
    z = conv2(x,y,mode='same')
    
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x,y, mode='constant', origin=origin)

    return z

def size_to_cpd(S,D):
    """
    Convert size of image to cycles per degree
    
    Parameters
    ----------    
    S : Size of image (same units)
    D : Distance from image (same units)
    
    Returns
    ----------
    V : visual angle 
    
    """
    return (2*np.arctan(S/(2*D)))
    
# Create white noise array
N=50 # Noise array dimension
A=1 # Noise amplitude
noise = np.random.rand(N,N)*A



# Gaussian Noise paramerers
GCenter=[0,0]
Gconst=0.5 # Magnitude of amplitude filtering (changes gaussian zero point)

# First gaussian filter
cutoff_f1 = 0.1 # < pi/10
gamma1 = 1/(2*np.pi*cutoff_f1) #minimum gamma == 0.5
Gamp1 = 1/(2*np.pi*gamma1)
filtr1 = Gaussian2D([0,0],Gamp1,gamma1,Gconst)
# Second gaussian filter
cutoff_f2 = 0.5 # < pi/10
gamma2 = 1/(2*np.pi*cutoff_f2) #minimum gamma == 0.5
Gamp2 = 1/(2*np.pi*gamma2)
filtr2 = Gaussian2D([0,0],Gamp2,gamma2,Gconst)

# Pad second filter with zeroes to match size of first filter 
empty = np.zeros(filtr1.shape)
d_0 = (filtr1.shape[0]-filtr2.shape[0]) / 2
d_1 = (filtr1.shape[0]-filtr2.shape[0]) / 2
empty[d_0:-d_0,d_0:-d_0]= filtr2
filtr2=empty

# Convolve Difference of Gaussian Output with noise
filtr_out = filtr1-filtr2 
noise_out = conv2(noise, filtr_out, mode='same')

# Convert to cpd
# noise_out = size_to_cpd(noise_out,0.1)
# noise = size_to_cpd(noise,0.1)

# Take Fourier transform of band pass filtered noise

fft_filtr1=np.fft.fftshift(np.fft.fft2(filtr1))
fft_filtr2=np.fft.fftshift(np.fft.fft2(filtr2))
fft_filtr_out=np.fft.fftshift(np.fft.fft2(filtr_out))
fft_noise=np.fft.fftshift(np.fft.fft2(noise))
fft_noise_filtr=np.fft.fftshift(np.fft.fft2(noise_out))

# Frequency space
N_filt = filtr1.shape[0]
nquist_noise = size_to_cpd(N/2,1)
nquist_filt = size_to_cpd(N_filt/2,1)
freq_noise = np.linspace(-nquist_noise,nquist_noise,N) # noise frequency space
freq_filtr = np.linspace(-nquist_filt,nquist_filt,N_filt) # filter frequency space

# Plot 1D Fourier outputs
plt.figure(1)
plt.subplot(5,1,1)
plt.plot(freq_filtr[N_filt/2:N_filt-1],np.abs(fft_filtr1[N_filt/2:N_filt-1, N_filt/2]))
plt.title('filtr1')
plt.subplot(5,1,2)
plt.plot(freq_filtr[N_filt/2:N_filt-1],np.abs(fft_filtr2[N_filt/2:N_filt-1, N_filt/2]))
plt.title('filtr2')
plt.subplot(5,1,3)
plt.plot(freq_filtr[N_filt/2:N_filt-1],np.abs(fft_filtr_out[N_filt/2:N_filt-1, N_filt/2]))
plt.title('D.O.G. Spectrum')
plt.subplot(5,1,4)
plt.plot(freq_noise[(N/2)+1:N],np.abs(fft_noise[(N/2)+1:N,N/2]))
#plt.ylim([0,np.max(np.abs(fft_noise[0:N/2,N/2]))])
plt.title('Noise Spectrum')
plt.subplot(5,1,5)
plt.plot(freq_noise[(N/2)+1:N],np.abs(fft_noise_filtr[(N/2)+1:N,N/2]))
#plt.ylim([0,np.max(np.abs(fft_noise[0:N/2,N/2]))])
plt.title('Filtered Noise Spectrum')


## Plot image outputs
plt.figure(2)
plt.subplot(2,1,1)
plt.imshow(noise,cmap='gray')
plt.title('noise')
plt.subplot(2,1,2)
plt.imshow(noise_out,cmap='gray')
plt.title('filtered noise')

plt.imshow(filtr_out,cmap='gray')
# convolve with ODOG for limits
# min max ranges of freq
# power of noise (contrast)
#return noise
    
#def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
#    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
#    f = np.zeros(samples)
#    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
#    f[idx] = 1
#    return fftnoise(f)
    
    
