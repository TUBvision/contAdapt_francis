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
direction = 's'
typ = 'diffuse'
contrast = 0.2
noise = 0
timeCount = 0
for time in np.arange(startT,stopT,stepT):
    
    # change adaptor color with each time step to produce flicker
    adaptorColorChange = -gray # black
    if np.mod(timeCount, 2)== 0:
        adaptorColorChange=gray        
    
    startInputImage = np.ones((i_x, i_y))*gray
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
    
    # Add noise
    #if noise[0] > 0:
    #    mask1=np.load("{0}{1}{2}{3}{4}{5}{6}".format('/home/will/Documents/noisemasks/noise',noise[1],'_',noise[2],'ppd_',noise[0],'0_5.npy'))
    #    startInputImage=(mask1[0:200,0:200]*255)+startInputImage    
              
    # Convert RGB input image to red-green, blue-yellow, white-black coordinates
    inputImage = np.zeros((200,200,3))
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
resultsDirectory= "/home/will/Documents/Git_Repository/contAdapt_francis/Code/Image_Outputs"
for i in np.arange(1,N):
    if i < 10:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All0',i,".png"))).convert('L'))/255.)
    else:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All',i,".png"))).convert('L'))/255.)

filename = "{0}{1}{2}{3}".format(resultsDirectory,"/",contrast,"_diffuse_s.gif")

imageio.mimsave(filename, images,duration=0.1)

"""
IDEAS FOR IMPROVEMENT
- Don't save images into files, just into an array and then compile into GIF
- General cleaning up and labelling of parameters
"""