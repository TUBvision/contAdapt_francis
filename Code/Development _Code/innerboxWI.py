# -*- coding: utf-8 -*-
"""
Creating inner patches to whitesillusion stimuli
"""
import numpy as np
import  whitesillusion as WI
import matplotlib.pylab as plt

shape=(2,2)
ppd=100
contrast=0.1
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
    stim[A, LEFT ] = 0.5*(L_1+(L_1*((A-y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A+y_pos)/p))) + 0.05
    stim[A, RIGHT] = 0.5*(L_0+(L_0*((A-y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A+y_pos)/p))) - 0.05

# Lower diffuse edges
for A in np.arange(-y_pos+p,-y_pos-p,-1):
    stim[A, LEFT]  = 0.5*(L_0+(L_0*((A+y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A-y_pos)/p)))# + 0.1
    stim[A, RIGHT] = 0.5*(L_1+(L_1*((A+y_pos)/p))) + 0.5*(mean_lum+(mean_lum*((-A-y_pos)/p)))# - 0.1

    
#brd=patch_height/3. #step border for innner check
#inc = contrast/4 #incremental step difference
#stim[y_pos+brd: -y_pos-brd, stim.shape[1] / 2 - (sep + 1) * half_cycle +brd: stim.shape[1] / 2 - sep * half_cycle-brd] = mean_lum+inc
#stim[y_pos+brd: -y_pos-brd, stim.shape[1] / 2 + sep * half_cycle +brd: stim.shape[1] / 2 + (sep + 1) * half_cycle-brd] = mean_lum+inc
#
plt.imshow(stim[:,:],cmap='gray',vmax=1.0,vmin=0)
