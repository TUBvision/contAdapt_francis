# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 01:49:52 2015

@author: Will
"""
import numpy as np

def CenterSurround2D(GCenter, Gamp, Ggamma, Samp, Sgamma):
    
    Gconst =2*np.log(2);
    new_theta = np.sqrt(Gconst^(-1))*Ggamma
    new_theta2 = np.sqrt(Gconst**(-1))*Sgamma
    if new_theta < .4 or new_theta2 < .4:
        disp('kernel is too small!')
        return 
    SizeHalf = floor(9*max(new_theta, new_theta2))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    
    a=-(1/2)*(Ggamma)**(-2)*Gconst*((x+GCenter[1])**2+(y+GCenter[2])**2)
    b=-(1/2)*(Sgamma)**(-2)*Gconst*((x+GCenter[1])**2+(y+GCenter[2])**2)
    GKernel = Gamp*np.exp(a) - Samp*np.exp(b)
    
    # normalize positive and negative parts
    posF = GKernel
    posF(posF<0)=0
    posnorm = np.sum(np.sum(posF))
    posF = posF/posnorm
    
    negF = GKernel
    negF(negF>0)=0
    negnorm = np.sum(np.sum(np.abs(negF)))
    negF = negF/negnorm
            
    GKernel = posF + negF
    # normalize full kernel
    # normalizer = sum(sum( GKernel.*GKernel ) );
    # GKernel = GKernel/sqrt(normalizer);
    return GKernel