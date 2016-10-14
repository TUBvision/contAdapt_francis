# -*- coding: utf-8 -*-
"""
This is a runfile for the structCAN model - Francis-Grossberg Model applied to 
common pyscophysical visual stimuli, extended for White's Illusion.

Parameters 
-------------------------------------------------------------------------------
Default:
------------------------------------------------------------------------------
evaluate(condition, patch_h = 1, direction = 'v', noise = 0, diffuse = 'n',
         contrast_f = 0.05, stopTime = 8 ,testOnset = 6)
-------------------------------------------------------------------------------        
direction  - determines the direction of the adapters on White's Illusion
diffuse    - determines whether the edges in WI are diffuse 
patch_H    - height of test patch in WI
noise mask - Options: [0.11,0.19,0.33,0.58,1.00,1.73,3.00,5.20,9.00] (0 if none)
contrast_f - contrast between dark and light stripes
-------------------------------------------------------------------------------
"""
import os
from PIL import Image
import imageio
import numpy as np
import CANNEM

# Set current folder to save images into
resultsDirectory= "/home/will/gitrepos/contAdaptTranslation/Code/Image_Outputs"
os.chdir(resultsDirectory)

# Run model
inst=CANNEM.base()
t_S=8
t_O=6
inst.evaluate(11,patch_h=0.25,direction='s', noise=0, diffuse='n', contrast_f=0.2,stopTime=t_S,testOnset=t_O)

# Compile images into GIF
N = t_S*10 # number of images
images=[]
for i in np.arange(1,N):
    if i < 10:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All0',i,".png"))).convert('L'))/255.)
    else:
        images.append(np.array(Image.open(("{0}{1}{2}{3}".format(resultsDirectory,'/All',i,".png"))).convert('L'))/255.)

filename = "{0}{1}".format(resultsDirectory,"/test.gif")

imageio.mimsave(filename, images,duration=0.1)



"""
TO DO
----------------
Define duration of stimuli above (onset etc)
Explicit oise mask creation
Define direction of diffuse edges in WI
Contrast only works for 0.2 ........ otherwise incorrect stimuli

"""










"""
-----------------------------
Alternative Gif making method
-----------------------------
From the commnand line, go to the directory with the images e.g.

"cd /home/will/gitrepos/contAdaptTranslation/Code/Image_Outputs"

and type

"convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif"

The gif will be generate in the current directory
"""