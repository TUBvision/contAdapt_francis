# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:20:45 2016

@author: will
"""
import os
os.chdir("/home/will/gitrepos/contAdaptTranslation")
import structCAN
mst=structCAN.CANNEM()
mst.evaluate(11,patch_h=0.25,direction='s',noise=3.00)

"""
noise mask - Options: [0.11,0.19,0.33,0.58,1.00,1.73,3.00,5.20,9.00] (0 if none)


From the commnand line, go to the directory with the images and type
convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif
"""