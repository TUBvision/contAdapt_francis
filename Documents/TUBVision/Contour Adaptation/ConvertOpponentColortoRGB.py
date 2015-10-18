# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 01:46:32 2015

@author: Will
"""
import nunpy as np

def ConvertOpponentColortoRGB(rg, by, wb, gray, max):

    # This function converts an array of color opponent values into RGB values
    arraySize = rg.shape;   
    rgb = zeros(arraySize[1], arraySize[2], 3);
    # green
    rgb[:,:,2] = gray*(3*wb - 3*rg - 2*by)/(3*max)
    # red
    rgb[:,:,1] = gray*2*rg/max + rgb[:,:,2]
    # blue
    rgb[:,:,3] = gray*2*by/max + (rgb[:,:,1] + rgb[:,:,2] )/2
    
    rgb = rgb + gray
    
    rgb(rgb<0) = 0
    rgb(rgb>255) = 255
    return rgb