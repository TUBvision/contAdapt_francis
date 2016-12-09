# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 15:45:23 2016

@author: Will
"""

from whitesillusion import degrees_to_pixels
import numpy as np
import matplotlib.pylab as plt

bg=(50,170) #the  background value of the stimulus. Default is 0.
            # 171cd/m^2 vs 51 cd/m^2
ppd=30
ssf=5
shape = (10,20) #Size of stimuli in degrees of visual angle. (y,x)

radii = (1.5,0.7) # Dimensions square and inner circle (diameter, radius)
square =  60 # gray values of rings (outer, inner)
                   # 58.2cd/m^2 vs 65-104cd/m^2
ellipse = np.linspace(65,104,7)
                    
# create stimulus at 5 times size to allow for supersampling antialiasing
stim = np.ones(degrees_to_pixels(np.array(shape), ppd).astype(int) * ssf)
stim[:,0:np.int(stim.shape[1]/2)]=bg[0]
stim[:,np.int(stim.shape[1]/2):-1]=bg[1]

# intermediate shading between two regions
width =100 # width of gradient 
for x in np.arange((stim.shape[1]/2)-width,(stim.shape[1]/2)+width,1):
    stim[:,x]=((bg[1]-bg[0])/(2.*width))*x - (stim.shape[1]/2.-width)/2. -142 + bg[0]

# Centrepoints of 2 sides
a=(np.int(stim.shape[1]/4.),np.int(stim.shape[1]*3/4.)) # x co-ordinates 
b=np.int(stim.shape[0]/2.)                              # y co ordinate 

# Draw the two square outer test patches
radii = degrees_to_pixels(np.array(radii), ppd) * ssf
radii = radii.astype(np.int64)
stim[b - radii[0]:b + radii[0], a[0] - radii[0]:a[0] + radii[0]] = square
stim[b - radii[0]:b + radii[0], a[1] - radii[0]:a[1] + radii[0]] = square

# Draw circles in the centre of the squares
N=1000
for t in np.linspace(-np.pi,np.pi,N):
    for r in np.linspace(0,radii[1],N):
        stim[np.int(b + r*np.sin(t)),np.int(a[0] + r*np.cos(t))]=ellipse[7]
        stim[np.int(b + r*np.sin(t)),np.int(a[1] + r*np.cos(t))]=ellipse[7]

# compute distance from center of array for every point, cap at 1.0
#x = np.linspace(a[0]-radii[1],a[0]+radii[1], 2*radii[1])
#y = np.linspace(b-radii[1],b+radii[1], 2*radii[1])
#Dist = np.sqrt((x[np.newaxis, :]-b) ** 2 + (y[:, np.newaxis]-a[0]) ** 2)
#
##for radius, value in zip(radii, values):
#stim[Dist < radii[1]] = values[1]

# downsample the stimulus by local averaging along rows and columns
#sampler = resize_array(np.eye(stim.shape[0] / ssf), (1, ssf))
#a = np.dot(sampler, np.dot(stim, sampler.T)) / ssf ** 2
    
plt.imshow(stim,cmap='gray',vmax=255)

#stimulus = np.zeros((7,stim.shape[0],stim.shape[1]))
#stimulus[7,:,:] = stim