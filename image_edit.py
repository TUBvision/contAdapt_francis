# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:09:08 2015

@author: will
"""

import numpy as np
import math
from scipy.ndimage.filters import convolve

def CenterSurround2D(GCenter, Gamp, Ggamma, Samp, Sgamma):
    
    Gconst =2*np.log(2);
    new_theta = np.sqrt(Gconst^(-1))*Ggamma
    new_theta2 = np.sqrt(Gconst**(-1))*Sgamma
    if new_theta < .4 or new_theta2 < .4:
        print('kernel is too small!')
        return 
    SizeHalf = math.floor(9*max(new_theta, new_theta2))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    
    a=-(1/2)*(Ggamma)**(-2)*Gconst*((x+GCenter[1])**2+(y+GCenter[2])**2)
    b=-(1/2)*(Sgamma)**(-2)*Gconst*((x+GCenter[1])**2+(y+GCenter[2])**2)
    GKernel = Gamp*np.exp(a) - Samp*np.exp(b)
    
    # normalize positive and negative parts
    posF = GKernel
    posF[posF<0]=0
    posnorm = np.sum(np.sum(posF))
    posF = posF/posnorm
    
    negF = GKernel
    negF[negF>0]=0
    negnorm = np.sum(np.sum(np.abs(negF)))
    negF = negF/negnorm
            
    GKernel = posF + negF
    # normalize full kernel
    # normalizer = sum(sum( GKernel.*GKernel ) );
    # GKernel = GKernel/sqrt(normalizer);
    return GKernel

def ConvertOpponentColortoRGB(rg, by, wb, gray, max):
    # This function converts an array of color opponent values into RGB values
    arraySize = rg.shape   
    rgb = np.zeros((arraySize[0], arraySize[1], 3))
    rgb[:,:,1] = gray*(3*wb - 3*rg - 2*by)/(3*max) # green
    rgb[:,:,0] = gray*2*rg/max + rgb[:,:,1] # red
    rgb[:,:,2] = gray*2*by/max + (rgb[:,:,0] + rgb[:,:,1] )/2 # blue
    rgb = rgb + gray
    rgb[rgb<0] = 0
    rgb[rgb>255] = 255
    return rgb
    
def  ConvertRGBtoOpponentColor(rgb, gray):
    # This function converts an array of RGB values into color opponent values
    rgb = rgb - gray
    rgb[rgb>127] = 127
    wb = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/ (3*gray) 
    rg = (rgb[:,:,0] - rgb[:,:,1])/(2*gray) 
    by = (rgb[:,:,2] - (rgb[:,:,0] + rgb[:,:,1])/2)/(2*gray) 
    return [rg, by,wb]
    
def im_cropping (*args):  
    InputMatrix = args[0]
    nPixCut = args[1]
    [x, y] = InputMatrix.shape
    StimX = x - 2 * nPixCut
    StimY = y - 2 * nPixCut
    # crop the actual stimulus area 
    output_matrix = InputMatrix[(nPixCut):(nPixCut+StimX), (nPixCut):(nPixCut+StimY)] 
    return output_matrix

def im_padding (*args):
    InputMatrix = args[0]
    PaddingSize = args[1]
    PaddingColor = args[2]
    [x, y] = InputMatrix.shape
    PaddedMatrix = np.ones((x+PaddingSize*2, y+PaddingSize*2))*PaddingColor
    a = PaddingSize # +1 removed as splint
    b= PaddingSize+x
    c = PaddingSize # +1 remove as splint
    d = PaddingSize+y
    PaddedMatrix[a:b,c:d]= InputMatrix
    return PaddedMatrix

def Gaussian2D(GCenter, Gamp, Ggamma,Gconst): #new_theta > 0.4:
    new_theta = math.sqrt(Gconst**-1)*Ggamma
    SizeHalf = np.int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    part1=(x+GCenter[0])**2+(y+GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)
    return GKernel
    
def floatingpointtointeger(decimal_places,value):
    first = np.float(10**decimal_places)
    second = 0.5**10**-decimal_places
    value = np.int((value * first ) + second) / first 
    return value
    
def conv2(x,y,mode='same'):
    """
    Emulate the function conv2 from Mathworks.
    Usage:
    z = conv2(x,y,mode='same')
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x,y, mode='constant', origin=origin)

    return z
