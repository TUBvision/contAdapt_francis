# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:40:25 2016

@author: Will

Investigating the spatial frequency statistics within an image....
In development only first ideas.
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Import jpg image or use square wave stimulus
filename = "adelson"
im = Image.open(("{0}{1}{2}".format("C:\Users\Will\Documents\gitrepos\contAdapt_francis\Documents\\" ,filename,".png"))).convert('L')

#plt.imshow(im,cmap='gray')

# Take Fourier of each component
fft_im=np.fft.fftshift(np.fft.fft2(im))
#threshold_fft = 20000
#fft_im[fft_im>threshold_fft]=0

#plt.imshow(np.abs(fft_im),cmap='gray')

# Summate radial orientations
N=fft_im.shape[0] # Needs to be converted into SF scale

# Pool orientations into array of polar radiis
count = 0
count2 = 0
points = 100 # across a radius
orientations = 360
fft_im_sum = np.zeros((points,orientations)) + 0j
for theta in np.linspace(0,np.pi*2,orientations):
    for R in np.linspace(0,(N/2)-1,points):
        x = R*np.cos(theta)+N/2
        y = R*np.sin(theta)+N/2
        fft_im_sum[count,count2]= fft_im[x,y]
        count =+ 1
    count2 =+ 1
        
# Summate Orientations
fft_im_sum = np.sum(fft_im_sum,axis=1)

fft_im_abs=np.abs(fft_im)
