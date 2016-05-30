# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:48:03 2016

@author: will
"""
import whitesillusion as wi
import image_edit
import math
import numpy as np

gray=127
# create whites illusion stimulus
direction = 'v'
patch_h = 0.25
stim, mask_dark, mask_bright = wi.evaluate(patch_h,direction)
i_x=stim.shape[0]
i_y=stim.shape[0]
stim=stim.T

inputImage = np.zeros((i_x, i_y, 3))
inputImage[:,:,0] = stim
inputImage[:,:,1] = stim
inputImage[:,:,2] = stim
[rg, by, wb] = image_edit.ConvertRGBtoOpponentColor(inputImage, gray)
          
# padding
PaddingSize = math.floor(np.max(wb.shape)/2) 
PaddingColor = wb[0,0]
wb2 = image_edit.im_padding(wb, PaddingSize, PaddingColor)

# orientation filters
# number of polarities (4 means horizontal and vertical orientations)
K = 4  
# number of orientations for simple cells
nOrient = K/2  
# boundaries are shifted relative to image plane(boundaries are between pixels)
orientationShift = 0.5  

gamma = 1.75
G = image_edit.Gaussian2D([0+orientationShift,0+orientationShift], 1, gamma, 2)
F = np.zeros((G.shape[0],G.shape[1],4))

for k in np.arange(0,K):
    m = np.sin((2*np.pi*(k+1))/K)
    n = np.cos((2*np.pi*(k+1))/K)
    H =image_edit.Gaussian2D([m+orientationShift,n+orientationShift],1,gamma,2)
    F[:,:,k] = G - H
    
    # normalize positive and negative parts
    posF = F[:,:,k]
    (posF>90).choose(posF,0)
    posnorm = np.sum(np.sum(posF*posF))
    posF = posF/np.sqrt(posnorm)
    
    negF = F[:,:,k]
    (negF>90).choose(negF,0)
    negnorm = np.sum(np.sum(negF*negF))
    negF = negF/np.sqrt(negnorm)
    
    F[:,:,k] = posF + negF
    
    # normalize full kernel
    normalizer = np.sum(np.sum( F[:,:,k]*F[:,:,k] ) )
    F[:,:,k] = F[:,:,k]/np.sqrt(normalizer)


#convolve orientation filters with White's Illusion

y=np.zeros((i_x,i_y,K))
for i in range(K):
    Ini = np.abs(image_edit.conv2(wb2, F[:,:,i])) # convolve
    Ini = image_edit.im_cropping(Ini, PaddingSize)  # padding
    Ini[Ini<0] = 0                                  # half wave rectify
    y[:,:,i]=Ini
    

# pool across contrast polarity
inI=5
planeSize = y.shape

z1= np.zeros((planeSize[0], planeSize[1], nOrient))
for k in np.arange(0,K/2):
    z1[:,:,k] = y[:,:,k] + y[:,:,k+K/2]

# set upper limit for boundary activity
#boundaryUpperLimit=25;
#z1[z1>boundaryUpperLimit] = boundaryUpperLimit

# Add tonic input, inI, to boundaries
w1= np.zeros((planeSize[0], planeSize[1], nOrient))
for k in np.arange(0,nOrient):
    w1[:,:,k] = inI  + z1[:,:,k]
    
stim_out=w1




# Orientation inhibiton
stim2=stim_out
for i in range(i_x):
    if stim2[i,0,0] > np.min(stim_out): # boundary signal above background minimum
        # now dealing with edge signal
        # scan across other perpendicular orientation
        for j in range(i_y):
                if stim2[i,j,0] < np.max(stim_out) and stim2[i,j,0] > np.min(stim_out): # bound test patch border signal
                    # inhibit border signal                    
                    stim2[i,j,0]=stim_out[i,j,0]*0.9

#
## view imput image (Whites Illusion) and two orientation edge images
import matplotlib.pyplot as plt
fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(20,10))
ax1.imshow(stim,cmap='gray')
ax2.imshow(w1[:,:,0],cmap='gray')
ax3.imshow(stim2[:,:,0],cmap='gray')
ax4.imshow(stim2[:,:,1],cmap='gray')
##
#import scipy
#filename3 = "{0}/{1}{2}{3}".format('/home/will/gitrepos/contAdaptTranslation','WI',patch_h,'.png')
#scipy.misc.imsave(filename3,stim)
##plt.imshow(w1[:,:,0],cmap='gray')
##
##plt.imshow(stim_out[122:128,:,0])
##
#import matplotlib.pyplot as plt
#fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(10,5))
#ax1.plot(z1[150,:,0])
#ax2.plot(z1[125,:,0])
#ax3.plot(z1[50,:,0])
#ax4.plot(z1[75,:,0])
#
#
#
#plt.plot(z1[:,87,1])
#
#import matplotlib.pyplot as plt
#fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(10,5))
#ax1.plot(y[150,:,0])
#ax2.plot(y[150,:,2])
#ax3.plot(y[150,:,1])
#ax4.plot(y[150,:,3])
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(10,5))
ax1.imshow(y[:,:,0])
ax2.imshow(y[:,:,1])
ax3.imshow(y[:,:,2])
ax4.imshow(y[:,:,3])
#
#from mayavi import mlab
#mlab.contour3d(y[:,:,0])

#invent play variable
y_edit=y
y_edit[y_edit>0.12]=0
y_edit[y_edit<0.04]=0

np.max(y[50:150,50:150,0])-np.max(y[50:150,50:150,2])
np.max(y[:,:,1])-np.max(y[:,:,3])
np.max(y_edit[:,:,1])
np.max(y_edit[:,:,2])
np.max(y_edit[:,:,3])


# 3D Magnitude Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')          # 3d axes instance
x2 = np.arange(0, 200, 1)              # points in the x axis
y2 = np.arange(0, 200, 1)              # points in the y axis
X, Y = np.meshgrid(x2, y2)     
surf = ax.plot_surface(X[2:197,2:197], Y[2:197,2:197], y_edit[2:197,2:197,3],
                       rstride=2,           # row step size
                       cstride=2,           # column step size
                       cmap='jet',        # colour map
                       linewidth=0 ,        # wireframe line width
                       antialiased=False
                       )


# Difference of Guassian exploration
fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(10,5))
ax1.plot(F[:,11,0])
ax2.plot(F[11,:,1])
ax3.plot(F[:,11,2])
ax4.plot(F[11,:,3])