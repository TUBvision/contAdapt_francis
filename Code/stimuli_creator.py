# -*- coding: utf-8 -*-
"""
This code takes the stimuli creator component out of CANNEM for speed when
investigating new types of stimuli with adaptation - WI only
"""

import numpy as np
import whitesillusion as wi
import os
import scipy
import imageio
from PIL import Image

def  ConvertRGBtoOpponentColor(rgb, gray):
    """
    This function converts an array of RGB values into color opponent values
    
    Parameters
    -----------
    rgb : array_like
            Combined color channels into RGB
    gray : int
            integer value of gray 
            
    Returns
    -----------
    rg/by/wb : array_like
            Seperate RGB color channels of input image
   
    """
    rgb = rgb - gray
    rgb[rgb>127] = 127
    wb = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/ (3*gray) 
    rg = (rgb[:,:,0] - rgb[:,:,1])/(2*gray) 
    by = (rgb[:,:,2] - (rgb[:,:,0] + rgb[:,:,1])/2)/(2*gray) 
    return [rg, by,wb]

# Input parameters
adaptorColorChange=5
i_x = i_y = 200
testOnset = 6
stopT = 10
stepT=startT= 0.1
gray = 127
patch_h = 0.25
direction = 'checker_odd'
typ = 'norm'
contrast = 0.1
noise = 0
timeCount = 0

# Checker parameters
N = 100 # Size of checker box to stack
h = 4 # Number of checkers stacked
i_x=500
i_y=800

shape = (i_x,i_y)
ppd=5
g_val=[1.5,1.6,1.7] 
        
for time in np.arange(startT,stopT,stepT):
    
    # change adaptor color with each time step to produce flicker
    adaptorColorChange = -gray # black
    if np.mod(timeCount, 2)== 0:
        adaptorColorChange=gray        
    
    startInputImage = np.ones((i_x, i_y))*gray
    if direction == 'checker' or direction == 'checker_odd':
        off = np.ones((N,N))*gray*g_val[2]
        on=np.ones((N,N))*gray*g_val[0]
        # Stack checker boxes to generate checker board
        first=np.hstack((h*[on,off]))
        second=np.hstack((h*[off,on]))
        checker=np.vstack((first,second,first,second,first))
        # Add same gray checkers to centre points
        checker[2*N:3*N,2*N:3*N]=gray*g_val[1]
        checker[2*N:3*N,5*N:6*N]=gray*g_val[1]
        stim = checker
        i_x = stim.shape[0]
        i_y = stim.shape[1]
        mask_dark,mask_bright = wi.contours_white_bmmc(shape,100,1,2,mean_lum=gray,contour_width=1,patch_height=None,orientation=direction)
    else:
        stim, mask_dark, mask_bright = wi.evaluate(patch_h,direction,typ,contrast)
    
    if time< testOnset: # Show adaptors (mask)
        if adaptorColorChange == gray:
            startInputImage= mask_bright
        else:
            startInputImage= mask_dark
    else: # Show test stimuli
        startInputImage = stim


    # fixation markers
    startInputImage[i_x/2-2,i_y/2-2]=255
    startInputImage[i_x/2  ,i_y/2  ]=255
    startInputImage[i_x/2-2,i_y/2  ]=0
    startInputImage[i_x/2  ,i_y/2-2]=0

    # Convert RGB input image to red-green, blue-yellow, white-black coordinates
    inputImage = np.zeros((stim.shape[0],stim.shape[1],3))
    inputImage[:,:,0] = startInputImage
    inputImage[:,:,1] = startInputImage
    inputImage[:,:,2] = startInputImage
    out = ConvertRGBtoOpponentColor(inputImage, gray)
    rg=out[0]
    by=out[1]
    wb=out[2]
    
    # Create directory for results
    resultsDirectory = os.path.dirname("{0}/".format("Image_Outputs"))
    
    if os.path.exists(resultsDirectory)==0:
        os.mkdir(resultsDirectory)        
    
    # Make image of input, boundaries, and filled-in values to save as a png file
    thing = np.ones((i_x, i_y, 3))
    
    # Input image on left (Participant Image)
    thing[:,0:i_y,:]=inputImage/255
    
    # Write individual frame files (with leading zero if less than 10)
    if timeCount>=10:
        filename = "{0}/{1}{2}{3}".format(resultsDirectory, 'All',timeCount,'.png')
    else:
        filename = "{0}/{1}{2}{3}".format(resultsDirectory,'All0',timeCount,'.png')
    
    #Same image to file
    scipy.misc.imsave(filename, thing)
    
    timeCount=timeCount+1

# Compile images into GIF
N = stopT*10 # number of images
images=[]
resultsDirectory= "C:\Users\Will\Documents\gitrepos\contAdapt_francis\Code\Image_Outputs"
for i in np.arange(1,N-1):
    if i < 10:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All0',i,".png"))).convert('L'))/255.)
    else:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All',i,".png"))).convert('L'))/255.)

filename = "{0}{1}{2}{3}{4}{5}{6}{7}".format(resultsDirectory,"/",contrast,"_",typ,"_",direction,".gif")

imageio.mimsave(filename, images,duration=0.1)

"""
IDEAS FOR IMPROVEMENT
- Don't save images into files, just into an array and then compile into GIF
- General cleaning up and labelling of parameters
- Incompatibility between the different conditions, test crossover matching
"""