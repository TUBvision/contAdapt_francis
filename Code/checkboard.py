# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:31:25 2016

@author: Will
"""

import numpy as np
import matplotlib.pylab as plt

N=100
mean_lum=127
contrast=1
contour_width=3

off = np.zeros((N,N))
on=np.ones((N,N))*mean_lum
h=4
v=3
first=np.hstack((h*[on,off]))
second=np.hstack((h*[off,on]))
checker=np.vstack((first,second,first,second,first))

checker[40:60,100:120]=0.1


# Add adapters
shape = checker.shape
idx_mask = np.zeros(shape, dtype=bool)
mask_dark = np.ones(shape) * mean_lum
mask_bright = np.ones(shape) * mean_lum
bright = mean_lum * (1 + contrast)
dark = mean_lum * (1 - contrast)
offset = contour_width // 2
y_pos=[150,350,200,300]
x_pos=[300,200,500,600,150,350,450,650]
offset = 5
ph=55

# Vertical Bars
idx_mask[y_pos[0]-ph: y_pos[0]+ph, x_pos[0] - offset : x_pos[0] + offset] = True
idx_mask[y_pos[0]-ph: y_pos[0]+ph, x_pos[1] - offset : x_pos[1] + offset] = True
idx_mask[y_pos[0]-ph: y_pos[0]+ph, x_pos[2] - offset : x_pos[2] + offset] = True
idx_mask[y_pos[0]-ph: y_pos[0]+ph, x_pos[3] - offset : x_pos[3] + offset] = True

idx_mask[y_pos[1]-ph: y_pos[1]+ph, x_pos[0] - offset : x_pos[0] + offset] = True
idx_mask[y_pos[1]-ph: y_pos[1]+ph, x_pos[1] - offset : x_pos[1] + offset] = True
idx_mask[y_pos[1]-ph: y_pos[1]+ph, x_pos[2] - offset : x_pos[2] + offset] = True
idx_mask[y_pos[1]-ph: y_pos[1]+ph, x_pos[3] - offset : x_pos[3] + offset] = True

# Horizontal Bars
idx_mask[y_pos[2] - offset : y_pos[2] + offset, x_pos[4] -ph: x_pos[4] + ph] = True
idx_mask[y_pos[2] - offset : y_pos[2] + offset, x_pos[5] -ph: x_pos[5] + ph] = True
idx_mask[y_pos[3] - offset : y_pos[3] + offset, x_pos[4] -ph: x_pos[4] + ph] = True
idx_mask[y_pos[3] - offset : y_pos[3] + offset, x_pos[5] -ph: x_pos[5] + ph] = True

idx_mask[y_pos[2] - offset : y_pos[2] + offset, x_pos[6] -ph: x_pos[6] + ph] = True
idx_mask[y_pos[2] - offset : y_pos[2] + offset, x_pos[7] -ph: x_pos[7] + ph] = True
idx_mask[y_pos[3] - offset : y_pos[3] + offset, x_pos[6] -ph: x_pos[6] + ph] = True
idx_mask[y_pos[3] - offset : y_pos[3] + offset, x_pos[7] -ph: x_pos[7] + ph] = True

mask_dark[idx_mask] = dark
mask_bright[idx_mask] = bright

plt.figure(2)
plt.imshow(mask_bright,cmap='gray')