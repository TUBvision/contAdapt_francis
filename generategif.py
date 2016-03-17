# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:12:19 2016

@author: will
"""

from PIL import Image, ImageSequence
from images2gif import writeGif
import os, 
import numpy as np
timeStep=0.1
stopTime=8
# convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif

resultsDirectory= "/home/will/gitrepos/contAdaptTranslation/Condition"
for i in np.arange(timeStep,stopTime+timeStep,timeStep):
    i = "%02d" % (i*10,)
    filename = "{0}{1}{2}{3}".format(resultsDirectory,'/All',i,'.png')
    im = Image.open(filename)
    original_duration = im.info['duration']
    frames = [frame.copy() for frame in ImageSequence.Iterator(im)]    
    


writeGif("reverse_" + os.path.basename(filename), frames, duration=original_duration/1000.0, dither=0)