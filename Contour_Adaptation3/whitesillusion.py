# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:32:30 2016

@author: will

Create White's Stimulus
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
    return stim


def whites_illusion_bmcc(shape, ppd, contrast, frequency, mean_lum=.5,
        patch_height=None, start='high', sep=1):
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
    stim = square_wave(shape, ppd, contrast, frequency, mean_lum, 'full',
                        start)
    half_cycle = int(degrees_to_pixels(1. / frequency / 2, ppd) + .5)
    if patch_height is None:
        patch_height = stim.shape[0] // 3
    else:
        patch_height = degrees_to_pixels(patch_height, ppd)
    y_pos = (stim.shape[0] - patch_height) // 2
    stim[y_pos: -y_pos,
         stim.shape[1] / 2 - (sep + 1) * half_cycle:
            stim.shape[1] / 2 - sep * half_cycle] = mean_lum
    stim[y_pos: -y_pos,
         stim.shape[1] / 2 + sep * half_cycle:
            stim.shape[1] / 2 + (sep + 1) * half_cycle] = mean_lum
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
        idx_mask[y_pos: -y_pos,
                 x_pos[0] - offset : x_pos[0] + offset] = True
        idx_mask[y_pos: -y_pos,
                 x_pos[0] + hc - offset : x_pos[0] + hc + offset] = True
        idx_mask[y_pos: -y_pos,
                 x_pos[1] - offset : x_pos[1] + offset] = True
        idx_mask[y_pos: -y_pos,
                 x_pos[1] + hc - offset : x_pos[1] + hc + offset] = True
    elif orientation == 'horizontal':
        idx_mask[y_pos - offset : y_pos + offset,
                 x_pos[0] : x_pos[0] + hc] = True
        idx_mask[y_pos - offset : y_pos + offset,
                 x_pos[1] : x_pos[1] + hc] = True
        idx_mask[-y_pos - offset : -y_pos + offset,
                 x_pos[0] : x_pos[0] + hc] = True
        idx_mask[-y_pos - offset : -y_pos + offset,
                 x_pos[1] : x_pos[1] + hc] = True
    mask_dark[idx_mask] = dark
    mask_bright[idx_mask] = bright
    return (mask_dark, mask_bright)
    
#def evaluate():
gray=127
stim = whites_illusion_bmcc((2,2),100,0.5,2)*255
mask_dark_h,mask_bright_h = contours_white_bmmc((2,2),100,0.5,2,mean_lum=gray/2,contour_width=2,orientation='horizontal')
mask_dark_v,mask_bright_v = contours_white_bmmc((2,2),100,0.5,2,mean_lum=gray/2,contour_width=2,orientation='vertical')

mask_dark_both = mask_dark_h + mask_dark_v
mask_bright_both = mask_bright_h + mask_bright_v

#return stim, mask_dark_both, mask_bright_both
import matplotlib.pyplot as plt
fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(15,15))
ax1.imshow(stim,vmin=np.min(mask_dark),vmax=np.max(mask_bright),cmap= 'gray')
ax2.imshow(mask_dark,vmin=np.min(mask_dark),vmax=np.max(mask_bright),cmap= 'gray')
im=ax3.imshow(mask_bright,vmin=np.min(mask_dark),vmax=np.max(mask_bright),cmap= 'gray')

# Splint version of colorbar inclusion
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.412, 0.05, 0.2])
fig.colorbar(im, cax=cbar_ax)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10,10))
ax1.imshow(thing[:,:,0],vmin=np.min(mask_dark),vmax=np.max(mask_bright),cmap= 'gray')