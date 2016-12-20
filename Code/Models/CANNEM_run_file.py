# -*- coding: utf-8 -*-
"""
Author: Will Baker Morrison

This is a runfile for the CANNEM model - Francis-Grossberg Model applied to 
common pyschophysical visual stimuli, extended for White's Illusion.

Default : evaluate(condition, patch_h = 1, direction = 'v', noise = 0,
                   diffuse = 'n', contrast_f = 0.05, stopTime = 8 ,testOnset = 6)
                   
Parameters 
-------------------------------------------------------------------------------        
patch_h    - Height of test patch in WI {default = 0.25}
noise mask - Options: [0.11,0.19,0.33,0.58,1.00,1.73,3.00,5.20,9.00] (0 if none)
contrast_f - contrast between dark and light stripes
typ        - Type of test patch (White's Illusion only)

                string from : ['norm','diffuse','inner']
                - Normal equiluminant test patches
                - Test patch with diffuse horizontal edges
                - Test patch with incremental inner patch

direction  - Location of flashing adapters (White's Illusion only)

                string from : ['h','v','s','t','both']
                - Horizontal bars immediate edges
                - Vertical bars immediate edges
                - Vertical "surround suppression" edges
                - T-junction adapters
                - Horizontal and Vertical immediate edges
                
t_S        - Full length of output GIF
testOnset  - Onset of stimulus, post adaptation

-------------------------------------------------------------------------------
"""
import os
from PIL import Image
import imageio
import numpy as np
import CANNEM

# Length of simulation
t_S = 10
contrast = 0.1

# Set current folder to save images into
resultsDirectory= "/home/will/Documents/Git_Repository/contAdapt_francis/Code/Image_Outputs"
os.chdir(resultsDirectory)

# Run model
inst=CANNEM.base()
inst.evaluate(11, patch_h=0.25, direction='s', noise=0, typ='diffuse', contrast_f=contrast, stopTime=t_S, testOnset=6)

# Compile images into GIF
N = t_S*10 # number of images
images=[]
resultsDirectory= "/home/will/Documents/Git_Repository/contAdapt_francis/Code/Image_Outputs/Image_Outputs"
for i in np.arange(1,N):
    if i < 10:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All0',i,".png"))).convert('L'))/255.)
    else:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All',i,".png"))).convert('L'))/255.)

filename = "{0}{1}{2}{3}".format(resultsDirectory,"/",contrast,"_diffuse_s.gif")

imageio.mimsave(filename, images,duration=0.1)



"""
----------------------------------------------
Alternative, Non-automated Gif making method
----------------------------------------------
From the commnand line, go to the directory with the images e.g.

"cd /home/will/gitrepos/contAdaptTranslation/Code/Image_Outputs"

and type

"convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif"

The gif will be generate in the current directory
"""