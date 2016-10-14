# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:32:30 2016

@author: will

Create White's Stimulus

Taken from Torstens code and restructured for this application.
Only real addition is evaluate tool which controls the other premade functions.
https://github.com/TUBvision/betz2015_contour
"""
import numpy as np

def degrees_to_pixels(degrees, ppd):
    """
    convert degrees of visual angle to pixels, given the number of pixels in
    1deg of visual angle.
    Parameters
    ----------
    degrees : number or ndarray
              the degree values to be converted.
    ppd : number
          the number of pixels in the central 1 degree of visual angle.
    Returns
    -------
    pixels : number or ndarray
    """
    return np.tan(np.radians(degrees / 2.)) / np.tan(np.radians(.5)) * ppd


def square_wave(shape, ppd, contrast, frequency, mean_lum=.5, period='ignore',
        start='high'):
    """
    Create a horizontal square wave of given spatial frequency.
    Parameters
    ----------
    shape : tuple of 2 numbers
            The shape of the stimulus in degrees of visual angle. (y,x)
    ppd : number
          the number of pixels in one degree of visual angle
    contrast : number in [0,1]
               the contrast of the grating, defined as
               (max_luminance - min_luminance) / mean_luminance
    frequency : number
                the spatial frequency of the wave in cycles per degree
    mean_lum : number
               the mean luminance of the grating, i.e. (max_lum + min_lum) / 2.
               The average luminance of the actual stimulus can differ slightly
               from this value if the stimulus is not an integer of cycles big.
    period : string in ['ignore', 'full', 'half'] (optional)
             specifies if the period of the wave is taken into account when
             determining exact stimulus dimensions.
             'ignore' simply converts degrees to pixesl
             'full' rounds down to garuantee a full period
             'half' adds a half period to the size 'full' would yield.
             Default is 'ignore'.
    start : string in ['high', 'low'] (optional)
            specifies if the wave starts with a high or low value. Default is
            'high'.
    Returns
    -------
    stim : 2D ndarray
           the square wave stimulus
    """

    if not period in ['ignore', 'full', 'half']:
        raise TypeError('size not understood: %s' % period)
    if not start in ['high', 'low']:
        raise TypeError('start value not understood: %s' % start)
    if frequency > ppd / 2:
        raise ValueError('The frequency is limited to 1/2 cycle per pixel.')

    shape = degrees_to_pixels(np.array(shape), ppd).astype(int)
    pixels_per_cycle = int(degrees_to_pixels(1. / frequency / 2, ppd) + .5) * 2

    if period is 'full':
        shape[1] = shape[1] / pixels_per_cycle * pixels_per_cycle
    elif period is 'half':
        shape[1] = shape[1] / pixels_per_cycle * pixels_per_cycle + \
                                pixels_per_cycle / 2
    diff = type(mean_lum)(contrast * mean_lum)
    high = mean_lum + diff
    low = mean_lum - diff
    stim = np.ones(shape) * (low if start is 'high' else high)
    index = [i + j for i in range(pixels_per_cycle / 2)
                      for j in range(0, shape[1], pixels_per_cycle)
                      if i + j < shape[1]]
    stim[:, index] = low if start is 'low' else high
    return stim, diff


def whites_illusion_bmcc(shape, ppd, contrast, frequency, mean_lum=.5,
        patch_height=None, start='high', sep=1,diffuse='n'):
    """
    Create a version of White's illusion on a square wave, in the style used by
    Blakeslee and McCourt (1999).
    Parameters
    ----------
    shape : tuple of 2 numbers
            The shape of the stimulus in degrees of visual angle. (y,x)
    ppd : number
          the number of pixels in one degree of visual angle
    contrast : number in [0,1]
               the contrast of the grating, defined as
               (max_luminance - min_luminance) / mean_luminance
    frequency : number
                the spatial frequency of the wave in cycles per degree
    mean_lum : number
               the mean luminance of the grating, i.e. (max_lum + min_lum) / 2.
               The average luminance of the actual stimulus can differ slightly
               from this value if the stimulus is not an integer of cycles big.
    patch_height : number
                   the height of the gray patches, in degrees of visual ange
    start : string in ['high', 'low'] (optional)
            specifies if the wave starts with a high or low value. Default is
            'high'.
    sep : int (optional)
          the separation distance between the two test patches, measured in
          full grating cycles. Default is 1.
    Returns
    -------
    stim : 2D ndarray
           the stimulus
    References
    ----------
    Blakeslee B, McCourt ME (1999). A multiscale spatial filtering account of
    the White effect, simultaneous brightness contrast and grating induction.
    Vision research 39(26):4361-77.
    """
    stim,diff = square_wave(shape, ppd, contrast, frequency, mean_lum, 'full',
                        start)
    half_cycle = int(degrees_to_pixels(1. / frequency / 2, ppd) + .5)
    if patch_height is None:
        patch_height = stim.shape[0] // 3
    else:
        patch_height = degrees_to_pixels(patch_height, ppd)
    y_pos = (stim.shape[0] - patch_height) // 2

    if diffuse == 'n':
        stim[y_pos: -y_pos, stim.shape[1] / 2 - (sep + 1) * half_cycle: stim.shape[1] / 2 - sep * half_cycle] = mean_lum
        stim[y_pos: -y_pos, stim.shape[1] / 2 + sep * half_cycle: stim.shape[1] / 2 + (sep + 1) * half_cycle] = mean_lum
    
    elif diffuse == 'y':
        p = np.round(0.25*patch_height)
        L_0 = mean_lum+diff # maximum lightness
        L_1 = mean_lum-diff# minimum lightness
        
        # Array for holding x axis coordinates
        LEFT=np.arange(stim.shape[1] / 2 - (sep + 1) * half_cycle, stim.shape[1] / 2 - sep * half_cycle)
        RIGHT=np.arange(stim.shape[1] / 2 + sep * half_cycle, stim.shape[1] / 2 + (sep + 1) * half_cycle)
        
        # Place test patches
        stim[y_pos: -y_pos, LEFT] = mean_lum
        stim[y_pos: -y_pos, RIGHT] = mean_lum
        
        # Upper diffuse edges
        for A in np.arange(y_pos+p,y_pos-p,-1):
            stim[A, LEFT ] = 0.5*(L_1+(L_1*((A-y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A+y_pos)/p))) + 0.1
            stim[A, RIGHT] = 0.5*(L_0+(L_0*((A-y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A+y_pos)/p))) - 0.1
        
        # Lower diffuse edges
        for A in np.arange(-y_pos+p,-y_pos-p,-1):
            stim[A, LEFT]  = 0.5*(L_0+(L_0*((A+y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A-y_pos)/p)))# + 0.1
            stim[A, RIGHT] = 0.5*(L_1+(L_1*((A+y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A-y_pos)/p)))# - 0.1

    else:
        print "Incorrect diffuse type y/n only"
        
    return stim

def contours_white_bmmc(shape, ppd, contrast, frequency, mean_lum=.5,
        patch_height=None, sep=1, orientation='vertical', contour_width=6):
    """
    Create stimuli with contours masking either the vertical or the horizontal
    borders of the test patches in White's illusion (Blakeslee, McCourt
    version).
    Parameters
    ----------
    shape : tuple of 2 numbers
            The shape of the stimulus in degrees of visual angle. (y,x)
    ppd : number
          the number of pixels in one degree of visual angle
    contrast : number in [0,1]
               the contrast of dark vs bright contours, defined as
               (max_luminance - min_luminance) / (2 * mean_luminance)
    frequency : number
                the spatial frequency of the White's stimulus to be masked in
                cycles per degree
    mean_lum : number
               the background luminance of the masking stimuli.
    patch_height : number
                   the height of the gray patches to be masked, in degrees of
                   visual ange
    sep : int (optional)
          the separation distance between the two test patches, measured in
          full grating cycles. Default is 1.
    orientation : ['vertical', 'horizontal'] (optional)
                  the orientation of the border to be masked. Default is
                  'vertical'.
    contour_width : number
                     the width of the masking contour in pixels
    Returns
    -------
    masks : tuple of two 2D ndarrays
            the contour adaptation masks. masks[0] has dark contours, mask[1]
            has bright contours.
    """
    shape = degrees_to_pixels(np.array(shape), ppd).astype(int)
    pixels_per_cycle = int(degrees_to_pixels(1. / frequency / 2, ppd) + .5) * 2
    shape[1] = shape[1] // pixels_per_cycle * pixels_per_cycle
    # determine pixel width of individual grating bars (half cycle)
    hc = pixels_per_cycle // 2
    if patch_height is None:
        patch_height = shape[0] // 3
    else:
        patch_height = degrees_to_pixels(patch_height, ppd)
    y_pos = (shape[0] - patch_height) // 2
    x_pos = (shape[1] // 2 - (sep + 1) * hc,
             shape[1] // 2 + sep * hc)
    mask_dark = np.ones(shape) * mean_lum
    mask_bright = np.ones(shape) * mean_lum
    idx_mask = np.zeros(shape, dtype=bool)
    bright = mean_lum * (1 + contrast)
    dark = mean_lum * (1 - contrast)
    offset = contour_width // 2
    if orientation == 'vertical':
        idx_mask[y_pos: -y_pos, x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[y_pos: -y_pos, x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[y_pos: -y_pos, x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[y_pos: -y_pos, x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        # Add cross hatching       
        for n in np.arange(87,-125,-2):
            idx_mask[n   : n+1, x_pos[0] - offset]      = False
            idx_mask[n+1 : n+2, x_pos[0]         ]      = False
            idx_mask[n   : n-1, x_pos[0] + offset]      = False
            
            idx_mask[n   : n+1, x_pos[0] + hc - offset]      = False
            idx_mask[n+1 : n+2, x_pos[0] + hc        ]      = False
            idx_mask[n   : n-1, x_pos[0] + hc + offset]      = False
            
            idx_mask[n   : n+1, x_pos[1] - offset]      = False
            idx_mask[n+1 : n+2, x_pos[1]         ]      = False
            idx_mask[n   : n-1, x_pos[1] + offset]      = False
            
            idx_mask[n   : n+1, x_pos[1] + hc - offset]      = False
            idx_mask[n+1 : n+2, x_pos[1] + hc         ]      = False
            idx_mask[n   : n-1, x_pos[1] + hc + offset]      = False
        mask_dark[np.roll(idx_mask,1,axis=0)] = dark # shift y axis 1 to create inverse crosshatch
        mask_bright[idx_mask] = bright
        mask_dark=mask_dark+mask_bright
        mask_bright=np.fliplr(mask_dark)
       
    elif orientation == 'horizontal':
        idx_mask[y_pos - offset : y_pos + offset, x_pos[0] : x_pos[0] + hc] = True
        idx_mask[y_pos - offset : y_pos + offset, x_pos[1] : x_pos[1] + hc] = True
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[0] : x_pos[0] + hc] = True
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[1] : x_pos[1] + hc] = True
        mask_dark[idx_mask] = dark
        mask_bright[idx_mask] = bright
        
    elif orientation == 'sup_dep_short':
        idx_mask[ y_pos+patch_height:y_pos+patch_height+y_pos/2, x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[ y_pos+patch_height:y_pos+patch_height+y_pos/2, x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[ y_pos/2: y_pos                               , x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[ y_pos/2: y_pos                               , x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[ y_pos+patch_height:y_pos+patch_height+y_pos/2, x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[ y_pos+patch_height:y_pos+patch_height+y_pos/2, x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        idx_mask[y_pos/2: y_pos                                , x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[y_pos/2: y_pos                                , x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        mask_dark[idx_mask] = dark
        mask_bright[idx_mask] = bright
        
    elif orientation == 'sup_dep_long':        
        idx_mask[ y_pos+patch_height:-1                   , x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[ y_pos+patch_height:-1                   , x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[ 0: y_pos                                , x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[ 0: y_pos                                , x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[ y_pos+patch_height:-1                   , x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[ y_pos+patch_height:-1                   , x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        idx_mask[ 0: y_pos                                , x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[ 0: y_pos                                , x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        mask_dark[idx_mask] = dark
        mask_bright[idx_mask] = bright
        
    elif orientation == 'T-junction':
        a=5 #vertical length
        b=5#horizontal length
        
        #Top test patch edges
        idx_mask[y_pos-patch_height/2+a: -y_pos-patch_height/2-a, x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[y_pos-patch_height/2+a: -y_pos-patch_height/2-a, x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[y_pos-patch_height/2+a: -y_pos-patch_height/2-a, x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[y_pos-patch_height/2+a: -y_pos-patch_height/2-a, x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        idx_mask[y_pos - offset : y_pos + offset, x_pos[0] : x_pos[0] + hc] = True
        idx_mask[y_pos - offset : y_pos + offset, x_pos[1] : x_pos[1] + hc] = True
        #idx_mask[y_pos - offset : y_pos + offset, x_pos[0]+b : x_pos[0] + hc -b] = False
        #idx_mask[y_pos - offset : y_pos + offset, x_pos[1]+b : x_pos[1] + hc - b] = False
        
        # Splitting T-junctions
        idx_mask[y_pos - offset : y_pos + offset, x_pos[0] +1: x_pos[0] +b] = False
        idx_mask[y_pos - offset : y_pos + offset, x_pos[0] -b +hc: x_pos[0] +hc -1] = False
        idx_mask[y_pos - offset : y_pos + offset, x_pos[1] +1: x_pos[1] + b] = False
        idx_mask[y_pos - offset : y_pos + offset, x_pos[1] -b +hc: x_pos[1] +hc -1] = False
        
        #Bottom test patch edges
        idx_mask[y_pos+patch_height/2+a: -y_pos+patch_height/2-a, x_pos[0] - offset      : x_pos[0] + offset]      = True
        idx_mask[y_pos+patch_height/2+a: -y_pos+patch_height/2-a, x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[y_pos+patch_height/2+a: -y_pos+patch_height/2-a, x_pos[1] - offset      : x_pos[1] + offset]      = True
        idx_mask[y_pos+patch_height/2+a: -y_pos+patch_height/2-a, x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[0] : x_pos[0] + hc] = True
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[1] : x_pos[1] + hc] = True
        #idx_mask[-y_pos - offset : -y_pos + offset, x_pos[0]+b : x_pos[0] + hc -b] = False
        #idx_mask[-y_pos - offset : -y_pos + offset, x_pos[1]+b : x_pos[1] + hc - b] = False
        
        #Splitting T-junctions
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[0]+1 : x_pos[0] + b] = False
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[0] -b +hc: x_pos[0] +hc -1] = False
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[1]+1 : x_pos[1] + b] = False
        idx_mask[-y_pos - offset : -y_pos + offset, x_pos[1] -b +hc: x_pos[1] +hc -1] = False
        
        mask_dark[idx_mask] = dark
        mask_bright[idx_mask] = bright
    
    return (mask_dark, mask_bright)



def evaluate(patch_h,direction,diffuse,contrast_f):
    gray=127
    #contrast_f = 5/245 # contrast differences used by g.francis model dark/light
    stim = whites_illusion_bmcc((2,2),100,contrast_f,2,patch_height=patch_h,diffuse='y')*255
    if direction == 'h': # horizontal bars
        mask_dark,mask_bright = contours_white_bmmc((2,2),100,1,2,mean_lum=gray/2,contour_width=2,patch_height=patch_h,orientation='horizontal')
    elif direction == 'v': # vertical bars
        mask_dark,mask_bright = contours_white_bmmc((2,2),100,1,2,mean_lum=gray/2,contour_width=2,patch_height=patch_h,orientation='vertical')
    elif direction == 's':
        mask_dark,mask_bright = contours_white_bmmc((2,2),100,1,2,mean_lum=gray,contour_width=2,patch_height=patch_h,orientation='sup_dep_long')
    elif direction == 't':
        mask_dark,mask_bright = contours_white_bmmc((2,2),100,1,2,mean_lum=gray,contour_width=2,patch_height=patch_h,orientation='T-junction')
    elif direction == 'both': # both directions
        mask_dark = []
        mask_dark_h,mask_bright_h = contours_white_bmmc((2,2),100,1,2,mean_lum=gray/4,contour_width=2,patch_height=patch_h,orientation='horizontal')
        mask_dark_v,mask_bright_v = contours_white_bmmc((2,2),100,1,2,mean_lum=gray/4,contour_width=2,patch_height=patch_h,orientation='vertical')
        mask_dark = mask_dark_h + mask_dark_v
        mask_bright = mask_bright_h + mask_bright_v
    else:
        print "Incorrect reference"
    return stim, mask_dark, mask_bright


####### Testing Code for Printing Output for different cases #######

#stim,mask_dark_supdep,mask_bright_supdep=evaluate(0.25,'t')
#import matplotlib.pyplot as plt
#fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(10,3))
#ax1.imshow(stim,cmap='gray')
#ax2.imshow(mask_dark_supdep,cmap='gray')
#ax3.imshow(mask_bright_supdep,cmap='gray')

#stim,mask_dark_v,mask_bright_v=evaluate(0.25,'v')
#
#import matplotlib.pyplot as plt
##plt.imshow(stim,cmap='gray')
#fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(10,10))
#ax1.imshow(mask_bright_both,cmap='gray')
#ax2.imshow(mask_dark_both,cmap='gray')
#ax3.imshow(mask_bright_v,cmap='gray')
#ax4.imshow(mask_dark_v,cmap='gray')
# Splint version of colorbar inclusion
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.82, 0.412, 0.05, 0.2])
#fig.colorbar(im, cax=cbar_ax)
#
#fig, (ax1) = plt.subplots(ncols=1, figsize=(10,10))
#ax1.imshow(thing[:,:,0],vmin=np.min(mask_dark),vmax=np.max(mask_bright),cmap= 'gray')