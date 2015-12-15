# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 01:43:53 2015

@author: Will
"""

def  ConvertRGBtoOpponentColor(rgb, gray):

    # This function converts an array of RGB values into color opponent values
    rgb = rgb - gray
    rgb(rgb>127) = 127
    wb = (rgb[:,:,1] + rgb[:,:,2] + rgb[:,:,3])/ (3*gray) 
    rg = (rgb[:,:,1] - rgb[:,:,2])/(2*gray) 
    by = (rgb[:,:,3] - (rgb[:,:,1] + rgb[:,:,2])/2)/(2*gray) 
    return [rg, by,wb]