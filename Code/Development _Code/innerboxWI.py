# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:44:18 2016

@author: will
"""

#import  whitesillusion as WI
import matplotlib.pylab as plt

shape=(2,2)
ppd=100
contrast=0.2
frequency=2
patch_height=0.25
sep=1
mean_lum=.5

stim,diff = WI.square_wave(shape, ppd, contrast, frequency)

half_cycle = int(WI.degrees_to_pixels(1. / frequency / 2, ppd) + .5)
if patch_height is None:
    patch_height = stim.shape[0] // 3
else:
    patch_height = WI.degrees_to_pixels(patch_height, ppd)
y_pos = (stim.shape[0] - patch_height) // 2

stim[y_pos: -y_pos, stim.shape[1] / 2 - (sep + 1) * half_cycle: stim.shape[1] / 2 - sep * half_cycle] = mean_lum
stim[y_pos: -y_pos, stim.shape[1] / 2 + sep * half_cycle: stim.shape[1] / 2 + (sep + 1) * half_cycle] = mean_lum
    
brd=patch_height/3. #step border for innner check
inc = contrast/4 #incremental step difference
stim[y_pos+brd: -y_pos-brd, stim.shape[1] / 2 - (sep + 1) * half_cycle +brd: stim.shape[1] / 2 - sep * half_cycle-brd] = mean_lum+inc
stim[y_pos+brd: -y_pos-brd, stim.shape[1] / 2 + sep * half_cycle +brd: stim.shape[1] / 2 + (sep + 1) * half_cycle-brd] = mean_lum+inc

plt.imshow(stim[:,:],cmap='gray',vmax=1.0,vmin=0)
