# -*- coding: utf-8 -*-
"""
Ring noisemask generate for Fourier space filtering
"""
from scipy.ndimage.filters import convolve
import numpy as np
import matplotlib.pylab as plt

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
N=200 # Noise array dimension
A=1 # Noise amplitude
noise = np.random.rand(N,N)*A

# Create ring filter
a, b = N/2, N/2
r_1 = 50
r_2 = 20

y,x = np.ogrid[-a:N-a, -b:N-b]
mask_1 = x*x + y*y <= r_1*r_1
mask_2 = x*x + y*y <= r_2*r_2
mask = mask_1 - mask_2

# Take Fourier of each component
fft_noise=np.fft.fftshift(np.fft.fft2(noise))

# Apply ring mask to fft of noise
fft_noise[mask] = 0

# Threshold DC term (for plotting)
fft_plot = np.abs(fft_noise)
fft_plot[100,100]=0

# Inverse filtered image
noise_out=np.fft.ifft2(fft_noise)
noise_out[100,100]=noise_out[100,100]-np.mean(noise_out)

# Image plotting
plt.subplot(2,2,1)
plt.imshow(mask,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(noise,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(fft_plot,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(np.abs(noise_out),cmap='gray')

"""
Relate radius of the disc in Fourier space to CPD
-------------------------------------------------
Convert pixels per degree visual angle (p/d)
Convert cycles per degree visual angle (V=2*arctan(S/2d)) in radians
-------------------------------------------------
r=100 (ppi)
d=30
ppd = 2drtan(0.5) => 52.36
"""