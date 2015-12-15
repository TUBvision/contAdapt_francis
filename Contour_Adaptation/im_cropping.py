# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 01:34:33 2015

@author: Will
"""

def im_cropping (varargin):  
    InputMatrix = varargin[1]
    nPixCut = varargin[2]
    [x, y] = InputMatrix.shape
    StimX = x - 2 * nPixCut
    StimY = y - 2 * nPixCut
    # crop the actual stimulus area 
    output_matrix = InputMatrix[(nPixCut+1):(nPixCut+StimX), (nPixCut+1):(nPixCut+StimY)] 
    return output_matrix