# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:06:15 2016

@author: will
"""
import numpy as np
import scipy as sc
import random

# Component functions

def narrowband_noise(freq, mask_size=512, ppd=31.277, noiseRms=0.2, max_noise=0.4, min_noise=-0.4):
    
    """
    Description
    -----------
    bandpass filtered white noise, centered on zero, that can be added to a
    stimulus. based on white_stim_innoise.m, but reduced to producing the noise
    mask.
    
    Noise Limits
    ------------
    we need to set limits on the noise to avoid that stimulus plus noise
    leaves the range (0,1). The limits depend on the stimulus contrast we
    want to use. For stim contrast .2, the grating bars will be at .4 and
    .6, so the noise must not be larger than .4 in any direction.    
    
    
    Authors   T. Betz
              V. Salmela (white_stim_innoise.m)
              T. Peromaa (noise filtering code)
    Translation    W. Baker Morrison
    """
       
    # noisemask parameters
    frequency = 2./3. * freq / ppd * mask_size # low cut off spatial frequency in cycles/image width (sf bandwidth is 1 oct) (image width = noise mask)
    
    nn = 2**np.ceil(np.log2(mask_size))
    
    # Create noise mask
    f = FreqFilt(nn,frequency,frequency*2,'G','P','E','+')
    
    const = noiseRms*nn**2/(np.sqrt(np.sum(f**2)))
    
    # Create random noise mask
    max_val = 2
    min_val = -2
    
    # Create noisemask until the extreme values are within desired range
    while max_val > max_noise or min_val < min_noise:
        s = GenSpec2(nn)
        sf = s*f*const
        fsf = np.fft.fftshift(sf)
        Y = np.fft.ifft2(fsf)
        noisemask = np.real(Y)
        # we need to divide the noise mask by two to achieve the same contrast
        # level that Salmela and Laurinen call RMS contrast, because their RMS
        # contrast is normalized by mean luminance, which is 0.5
        noisemask = noisemask / 2
        if nn != mask_size:
            noisemask = noisemask[0:mask_size, 0:mask_size]

        max_val = np.max(noisemask[:])
        min_val = np.min(noisemask[:])
        
    return noisemask
    


def GenSpec2(n):
    """
    Parameters
    ----------
    n           - size of the spectrum (nxn), n should be a power of 2
    
    
    Returns
    ----------
    spectrum    - shifted 2D complex number spectrum [±n/2...-1,0,1,...(n/2-1)], pseudorandom white noise
                  DC = 0;
    A/2         - The amplitude of each (pos or neg) frequency component:   	
    A           - The amplitude of each (whole) frequency component: 		
    (A/2)^2     - The power of each (pos or neg) frequency component: 		
    A^2         - The power of each (whole) frequency component: 			
    A^2         - The power of DC-component:						      
    
    Author   T. Peromaa 
    Translation W. Baker Morrison
    
    """
    spectrum=np.zeros((n,n),dtype=np.complex128)
    row=np.zeros((1,n))
    column=np.zeros((n,1))
    Amp = 2	#5000
    HalfAmp = Amp/2
    #Power = Amp^2
    #Nyqv = n/2
    
    # quadrant 1
    Re = np.random.rand(n/2-1)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))	 
    quadrant1 = Re+Im*1j
        	
    # quadrant 2
    Re = np.random.rand(n/2-1)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))	
    quadrant2 = Re+Im*1j
    	
    # quadrant 3
    quadrant3 = list(reversed(quadrant2))
    quadrant3 = np.conj(quadrant3)
    
    # quadrant 4 
    quadrant4 = list(reversed(quadrant1)) 
    quadrant4 = np.conj(quadrant4)
    
    # Set quandrants into spectrum
    spectrum[1:n/2,1:n/2]     = quadrant1
    spectrum[1:n/2,n/2+1:n]   = quadrant2
    spectrum[n/2+1:n,1:n/2]   = quadrant3
    spectrum[n/2+1:n,n/2+1:n] = quadrant4
    
    # kŠsitellŠŠn - first line of the spectrum: f1=±n/2
    Re = np.random.rand(1,n)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))	
    row = Re+Im*1j
    apu = np.fliplr(row)
    row[n/2+1:n] = np.conj(apu[n/2:n-1]) 
    spectrum[0,:] = row
    
    # kŠsitellŠŠn f1=0
    Re = np.random.rand(1,n)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))	
    row = Re+Im*1j
    apu = np.fliplr(row)
    row[n/2+1:n] = np.conj(apu[n/2:n-1])
    spectrum[n/2,:]= row
    
    # kŠsitellŠŠn f2=0
    Re = np.random.rand(n,1)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))
    column = Re+Im*1j
    apu = np.flipud(column)
    column[n/2+1:n] = np.conj(apu[n/2:n-1])
    spectrum[:,n/2] = column.squeeze()
    
    # kŠsitellŠŠn f2=±n/2
    Re = np.random.rand(n,1)*Amp-HalfAmp
    Im = np.sqrt(HalfAmp**2-Re*Re)
    Im = Im*random.choice((-1,1))
    column = Re+Im*1j
    apu = np.flipud(column)
    column[n/2+2:n] = np.conj(apu[n/2+1:n-1])
    spectrum[:,1]= column.squeeze()
    
    # kŠsitellŠŠn separate components, which are not conjugate (phase=2¹ or Re = -HalfAmp?)
    # f1=f2=±n/2
    # f1=±n/2,f2=0
    # f1=0,f2=±n/2 
    # fi=0,f2=0
    	
    spectrum[0,0]       = -HalfAmp+0*1j
    spectrum[0,n/2]     = -HalfAmp+0*1j
    spectrum[n/2,0]     = -HalfAmp+0*1j
    spectrum[n/2,n/2]   = 0+0*1j
    	
    return spectrum

#f = FreqFilt(512.0,1.2,2.4,'G','P','E','+')
def FreqFilt(n,low,high,IG,PR,EO,OnOff):
    
    """
    PURPOSE:	Generated a spectrum of bandpass/reject filter (POWER, not amplitude)
    
    INPUT:  	
    n: 			size of the filter (should be the same as the image size and a power of 2)
            	low, high:	Frequency limits of the filter
    IG:			I = ideal: low & high included into the spectrum
              		G = Gaussian: low & high (1/2 power value)
    PR:			P = Pass
    				R = Reject
    EO:			E = even
    				O = odd	(does not make any sense in symmetrical filter)
    OnOff:		      + = on-center
    				- = off-center
         
    OUTPUT: 	
    Shifted complex number spectrum of a bandpass/reject filter

    Author      T. Peromaa 
    Translation W. Baker Morrison
    
    """
    # Check the input
    check = np.arange(2,12)
    check = 2**check
    if any(n==check) == False:
        print 'Filter size (n) should be a power of two'
        return
    if IG != 'I' and IG != 'G':
        print 'IG has only two possible values: I and G'
        return
    if PR != 'P' and PR != 'R':
        print 'PR has only two possible values: P and R'
        return
    if EO != 'E' and EO != 'O':
        print 'EO has only two possible values: E and O'
        return
    if OnOff != '+' and OnOff != '-':
        print 'OnOff has only two possible values: + and -'
        return
    
    if EO == 'O':
        print '€ Circularly symmetric ODD filter does not make sense...'
    		
    # Calculate u-frequency
    u = np.zeros((n,n))
    for j in np.arange(0,n):
        u[j,:] = np.ones((1,n))*(j-n/2-1)
    	
    # Calculate v-frequency
    v = np.zeros((n,n))
    for j in np.arange(0,n):
        v[:,j] = (np.ones((n,1))*(j-n/2-1)).squeeze()
    
    # Calculate 2D spatial frequency
    frequency = (u**2+v**2)**(1/2)
    del u
    del v
    	
    # Calculate requested center frequency and space constant (‰ width)
    center = (low+high)/2
    SC = center-low
    		
    # Calculate the distance of each 2D spatial frequency from the requested centre frequency
    distance = np.abs(center-frequency)
    
    # Generate the bandpass filter spectrum: either IDEAL or GAUSSIAN
    filter_out = np.zeros((n,n))
    	
    if IG == 'I':
    	 k = np.where(distance<=SC)	
    	 filter_out[k] = np.ones(np.size(k))
    else:
        # SC is transformed, because it refers to the 1/e and not to the 1/2 value point
        SC = 1/np.sqrt(np.log(2))*SC
        filter_out = np.exp(-(distance/SC)**2)	# Graham, Robson & Nachmias (1978)
        # SQRT is used, because the purpose is to filter POWER, not amplitude
        k = np.where(filter_out>0)
        filter_out[k] = (filter_out[k])**(1/2)
    
    # correct the conjugates in row 1: f1=±n/2	(f1+ is the correct one)
    CorrectRow = filter_out[0,n/2+1:n]
    filter_out[0,1:n/2] = list(reversed(CorrectRow))
    
    # correct the conjugates in column 1: f2=±n/2 (f2+ is the correct one)	
    CorrectCol = filter_out[n/2+1:n,0]
    filter_out[1:n/2,0] = list(reversed(CorrectCol))
    	
    # If the filter is REJECT, revert the values
    if (PR=='R'):
        filter_out = 1-filter_out

    # if the filter is ODD, Re=0, Im = gain and Im = -gain for the conjugates
    if EO == 'O':
    		RealPart = np.zeros((n,n))
    		ImagPart = filter_out
    		ImagPart[1:n/2,1:n/2] = -1*(ImagPart[1:n/2,1:n/2])			# quadrant: left, top
    		ImagPart[n/2+1:n,1:n/2] = -1*(ImagPart[n/2+1:n,1:n/2])		# quadrant: left, bottom
    		ImagPart[0,1:n/2] = -1*(ImagPart[0,1:n/2])    			# row 1
    		ImagPart[n/2,1:n/2] = -1*(ImagPart[n/2,1:n/2])			# row n/2+1
    		ImagPart[1:n/2,0] = -1*(ImagPart[1:n/2,0])				# column 1
    		ImagPart[1:n/2,n/2] = -1*(ImagPart[1:n/2,n/2])        	# column n/2+1
    		ImagPart[0,0] = 0
    		ImagPart[0,n/2] = 0
    		ImagPart[n/2,0] = 0
    		ImagPart[n/2,n/2] = 0;
    		filter_out = RealPart+ImagPart*1j

    
    # if the filter_out is off-type, change the sign of all values
    if OnOff == '-':
    	 filter_out = -1*filter_out
     
    return filter_out
    

# Originally called generate_noisemasks.m - Ref: Torsten Betz
""" 
generate 25 noise masks at all frequencies for model testing. only first 5
are required for psychophysical experiment.

"""
rang = np.arange(-np.log2(9),np.log2(9),np.log2(9)/4)
rang = np.round(2**(rang)*100)/100
#for noise_freq in np.round(2**(rang)*100)/100 :
noise_freq = rang[2]
if noise_freq > 2:
    noise_max = .43
    noise_min = -.43
else:
    noise_max = .4
    noise_min = -.4

ppd = 31.2770941620795
mask_size = 512
 
      
k = 2 #in range(25):
#noise = narrowband_noise(noise_freq, mask_size, ppd, .2, noise_max, noise_min)
#sc.save("..\noise\noise%i_%.3fppd_%.3f_%.3f.mat" % ( mask_size, ppd, noise_freq, k), 'noise')


freq=noise_freq
mask_size=512
ppd=31.277
noiseRms=0.2
max_noise=0.4
min_noise=-0.4
    
   
# noisemask parameters
frequency = 2./3. * freq / ppd * mask_size # low cut off spatial frequency in cycles/image width (sf bandwidth is 1 oct) (image width = noise mask)

nn = 2**np.ceil(np.log2(mask_size))

# Create noise mask
f = FreqFilt(nn,frequency,frequency*2,'G','P','E','+')

const = noiseRms*nn**2/(np.sqrt(np.sum(f**2)))

# Create random noise mask
max_val = 2
min_val = -2

values_min = [0]
values_max = [0]
# Create noisemask until the extreme values are within desired range
while max_val > max_noise or min_val < min_noise:
    s = GenSpec2(nn)
    sf = s*f*const
    fsf = np.fft.fftshift(sf)
    Y = np.fft.ifft2(fsf)
    noisemask = np.real(Y)
    # we need to divide the noise mask by two to achieve the same contrast
    # level that Salmela and Laurinen call RMS contrast, because their RMS
    # contrast is normalized by mean luminance, which is 0.5
    noisemask = noisemask / 2
    if nn != mask_size:
        noisemask = noisemask[0:mask_size, 0:mask_size]
    
    max_val = np.max(noisemask)
    min_val = np.min(noisemask)
    
    values_min.append(min_val)
    values_max.append(max_val)

