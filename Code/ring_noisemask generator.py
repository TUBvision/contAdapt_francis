# -*- coding: utf-8 -*-
"""
Ring noisemask generate for Fourier space filtering
"""
from scipy.ndimage.filters import convolve
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

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
N=300 # Noise array dimension in pixels
A=1 # Noise amplitude
noise = np.random.rand(N,N)*A

# Otherwise import jpg image 
filename = "zebra"
im = Image.open(("{0}{1}{2}".format("/home/will/Documents/Images/",filename,".png"))).convert('L')
# Resizing image (smaller) increases speed (but reduces accuracy)
f_reduce = 1 # reduction factor
arr = np.array(im.resize((N,N), Image.ANTIALIAS))
noise = arr

# Create ring filter [approx radius/3 for spatial freq]
a, b = N/2, N/2
r_1 = 20 #Outer radius of ring filter 
r_2 = 1 #Inner radius of ring filter

y,x = np.ogrid[-a:N-a, -b:N-b]
mask_1 = x*x + y*y <= r_1*r_1
mask_2 = x*x + y*y <= r_2*r_2
mask = mask_1 - mask_2

# Inverse mask
#mask=(mask*-1)+1

# Take Fourier of each component
fft_noise=np.fft.fftshift(np.fft.fft2(noise))

# Apply ring mask to fft of noise
fft_noise_un = fft_noise
fft_noise=fft_noise*mask

# Threshold DC term (for plotting)
fft_plot = np.abs(fft_noise)
fft_plot[N/2,N/2]=0
fft_plot_un = np.abs(fft_noise_un)
fft_plot_un[N/2,N/2]=0

# Inverse filtered image
noise_out=np.fft.ifft2(fft_noise)
#noise_out[N/2,N/2]=noise_out[N/2,N/2]-np.mean(noise_out) # Threshold DC term

# Convert into cycles per degree visual angle (cm)
S = 5. # size of image in cm
D = 15. # distance from image in cm
V = 2 *np.arctan(S/(2*D))

# Image plotting
#plt_limit = 10
plt.figure(1,figsize=[S,S]) # Width x Heigh in inches
plt.subplot(2,1,1)
plt.imshow(np.log(fft_plot_un),cmap='gray')#,extent=[-(N*V)/2,(N*V)/2,-(N*V)/2,(N*V)/2])
plt.title('Fourier unmasked')

plt.subplot(2,1,2)
plt.imshow(np.log(fft_plot),cmap='gray',extent=[-(N*V)/2,(N*V)/2,-(N*V)/2,(N*V)/2])
plt.title('Fourier masked')
plt.locator_params(axis='x',nbins=10)
plt.locator_params(axis='y',nbins=10)
#plt.xlim([-plt_limit,plt_limit])
#plt.ylim([-plt_limit,plt_limit])


plt.figure(2,figsize=[7,7])
plt.subplot(2,1,1)
plt.imshow(np.abs(noise),cmap='gray')
plt.title('Input')
plt.subplot(2,1,2)
plt.imshow(np.abs(noise_out),cmap='gray')
plt.title('Output')


"""
Relate radius of the disc in Fourier space to CPD
-------------------------------------------------
Convert pixels per degree visual angle (p/d)
Convert cycles per degree visual angle (V=2*arctan(S/2d)) in radians


Pixel per degree conversion
---------------------------
r=100 (ppi)
d=30
ppd = 2drtan(0.5) => 52.36

Pixel dimensions
---------------------------
UP= (22*25.4) / 1920 
HOR=(18*25.4) / 1200

To do
---------------------------
Include explicit CPD noise band input
"""
