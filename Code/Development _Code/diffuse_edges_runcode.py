# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:23:56 2016

@author: will
"""

#import whitesillusion as WI
import numpy as np
import matplotlib.pylab as plt

contrast_f = 0.2 
shape=(2,2)
ppd=100
f=2
patch_height=0.25
mean_lum=.5
diffuse ='y'
sep=1
start='high'




stim,diff = WI.square_wave(shape, ppd, contrast_f, f, mean_lum, 'full',start)
half_cycle = int(WI.degrees_to_pixels(1. / f / 2, ppd) + .5)
if patch_height is None:
    patch_height = stim.shape[0] // 3
else:
    patch_height = WI.degrees_to_pixels(patch_height, ppd)
y_pos = (stim.shape[0] - patch_height) // 2

plt.imshow(stim,cmap='gray',vmin=0,vmax=1)