# -*- coding: utf-8 -*-
"""
Translation G.Francis Gaussian Kernel Definition
"""
import numpy as np
import math

def Gaussian2D(GCenter, Gamp, Ggamma,Gconst):
    new_theta = np.sqrt(Gconst**-1)*Ggamma
    if new_theta < .4:
        print('kernel is too small!')
    SizeHalf = int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(range(-SizeHalf,SizeHalf+1), range(-SizeHalf,SizeHalf+1))
    part1=(x+GCenter[0])**2+(y+GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)    
    return GKernel
