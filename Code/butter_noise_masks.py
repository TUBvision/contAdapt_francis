# -*- coding: utf-8 -*-
"""
Butter band-pass filtered white noise for creation of noise masks to test
spatial frequency specificity of lightness models
"""

from scipy.signal import butter, lfilter,freqz
import numpy as np
import matplotlib.pylab as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
     y = lfilter(b, a, data)
     return y
    
"""
Spatial to frequency domain -> shift midpoint from x/2,y/2 to 0,0 with +/- x & y

Outer part of 2D Frequency spectrum represents edges
Inner part represents average pixel values

Ring would represent filter we need (band between low and high freq)
V=2*arctan(s/2d)
convert S into V?

In 2d frequency domain, how does the scale match visual angle scale?
X axis goes from -x/2 to x/2, max frequency is therefore 1/(x/2)?


With plotting: -Nquist:Nquist..... but may as well plot positive side of the
spectrum as it is symmetrical.

What is the maximum spatial frequency in the initial image? NYQUIST FREQ = N/2

- First convert into visual angle
- Then divide by one to get cycles per degree visual angle?

Then we have to convert pixel frequency to "cycles per degee visual angle"

"""

# Create white noise array
N=50 # Noise array dimension
A=1 # Noise amplitude
noise = np.random.rand(N,N)*A     
     
# Filter parameters     
fs = 1000.0 
lowcut = 50.0
highcut = 60.0

b,a = butter_bandpass(lowcut, highcut, fs)
w, h = freqz(b, a, worN=2000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h))


out = butter_bandpass_filter(noise,lowcut,highcut,fs)


plt.subplot(2,1,1)
plt.imshow(noise,cmap='gray')
plt.subplot(2,1,2)
plt.imshow(out,cmap='gray')