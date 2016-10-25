# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:24:04 2016

@author: will
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


# Create white noise array
N=50 # Noise array dimension
A=10 # Noise amplitude
noise = np.random.rand(N,N)*A

# Gaussian Noise paramerers
GCenter=[0,0]
Gconst=1

# First gaussian filter
cutoff_f1 = 0.05 # < pi/10
gamma1 = 1/(2*np.pi*cutoff_f1) #minimum gamma == 0.5
Gamp1 = 1/(2*np.pi*gamma1)
filtr1 = Gaussian2D([0,0],Gamp1,gamma1,Gconst)
# Second gaussian filter
cutoff_f2 = 0.04 # < pi/10
gamma2 = 1/(2*np.pi*cutoff_f2) #minimum gamma == 0.5
Gamp2 = 1/(2*np.pi*gamma2)
filtr2 = Gaussian2D([0,0],Gamp2,gamma2,Gconst)

# Convolve filters with noise
noise_filtr1 = conv2(noise, filtr1, mode='same')
noise_filtr2 = conv2(noise, filtr2, mode='same')

# Difference of Gaussian Output
noise_out = noise_filtr1- noise_filtr2

# Take Fourier transform of band pass filtered noise
fft_noise=np.fft.fftshift(np.fft.fft2(noise))
fft_filtr1=np.fft.fftshift(np.fft.fft2(filtr1))
fft_filtr2=np.fft.fftshift(np.fft.fft2(filtr2))
fft_noise_filtr=np.fft.fftshift(np.fft.fft2(noise_out))

# PLot Fourier outputs
plt.figure(1)
plt.subplot(4,2,1)
plt.plot(filtr1[:,filtr1.shape[1]/2])
plt.title('filter1')
plt.subplot(4,2,3)
plt.plot(filtr2[:,filtr2.shape[1]/2])
plt.title('filter2')
plt.subplot(4,2,5)
plt.plot(noise[:,N/2])
plt.title('noise')
plt.subplot(4,2,7)
plt.plot(noise_out[:,N/2])
plt.title('noise_filtr')


plt.subplot(4,2,2)
plt.plot(fft_filtr1[:,fft_filtr1.shape[1]/2])
plt.title('fft_filter1')
plt.subplot(4,2,4)
plt.plot(fft_filtr2[:,fft_filtr2.shape[1]/2])
plt.title('fft_filter2')
plt.subplot(4,2,6)
plt.plot(fft_noise[:,N/2])
plt.title('fft_noise')
plt.subplot(4,2,8)
plt.plot(fft_noise_filtr[:,N/2])
plt.title('fft_noise_filtr')


## Plot image outputs
plt.figure(2)
plt.subplot(4,2,1)
plt.imshow(filtr1,cmap='gray')
plt.title('filter1')
plt.subplot(4,2,3)
plt.imshow(filtr2,cmap='gray')
plt.title('filter2')
plt.subplot(4,2,5)
plt.imshow(noise,cmap='gray')
plt.title('noise')
plt.subplot(4,2,7)
plt.imshow(noise_out,cmap='gray')
plt.title('filtered noise')

plt.subplot(4,2,2)
plt.imshow(np.abs(fft_filtr1),cmap='gray')
plt.title('filter')
plt.subplot(4,2,4)
plt.imshow(np.abs(fft_filtr2),cmap='gray')
plt.title('filter2')
plt.subplot(4,2,6)
plt.imshow(np.abs(fft_noise),cmap='gray')
plt.title('noise')
plt.subplot(4,2,8)
plt.imshow(np.abs(fft_noise_filtr),cmap='gray')
plt.title('filtered noise')







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
    
    
