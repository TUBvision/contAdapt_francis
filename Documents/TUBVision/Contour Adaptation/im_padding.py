# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 01:32:17 2015

@author: Will
"""
import numpy as np

def im_padding (varargin):
    InputMatrix = varargin[1]
    PaddingSize = varargin[2]
    PaddingColor = varargin[3]
    
    [x, y] = InputMatrix.shape
    PaddedMatrix = np.zeros(x+PaddingSize*2, y+PaddingSize*2)
    PaddedMatrix[:,:] =  PaddingColor
    PaddedMatrix[PaddingSize+1:PaddingSize+x, PaddingSize+1:PaddingSize+y] = InputMatrix
    return PaddedMatrix

