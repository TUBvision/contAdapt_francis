# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:57:46 2015

@author: will
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:26:40 2015
Translated Python Version of Gregory Francis' contour adaptation
"""

# From the commnand line, go to the directory with the images and type
# convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif

import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import math
import os
import scipy
import time 

#import cv2
#import pandas as pd
#from numba import jit

#from scipy.ndimage.measurements import mean as labeled_mean
#import matplotlib.pyplot as plt

class contourAdaptation(object):


    """ 
    Model of contour adaptation translated from G.Francis Matlab Code
        
    basic use:
    contourAdaptation 
    
    """
    # initializer
    def __init__(self,condition):
                 self.startTeam=time.clock()
                 self.makeAnimatedGifs=0
                 self.gray = 127
                 self.shift = np.array([[-1,0],[1,0],[0,-1],[0, 1]])
                 self.shiftOrientation = [np.pi/2, np.pi/2, 0, 0]
                 self.Bshift1 = np.array([[-1, -1],[ 0, -1],[ -1, -1],[ -1, 0]])
                 self.Bshift2 = np.array([[-1,  0],[ 0,  0],[  0, -1],[  0, 0]])
                 self.i_x=200
                 self.i_y=200
                 self.C =[]
                 self.alpha =.5 
                 self.beta = 2                  
                 self.Agate = 20.0  
                 self.Bgate = 1.0
                 self.Cgate = 1.0
                 self.Rhogate = 0.007  
                 self.inI = 5
                 self.bThresh=9.5
                 self.adaptorColorChange = 0
                 self.adaptorSize = 0
                 self.testColor = 0
                 self.midPoint =[0,0]
                 self.middleadaptorSize=0
                 self.inneradaptorSize=0
                 self.K = 4  
                 self.nOrient = self.K/2
                 self.orientationShift = 0.5
                 self.gamma = 1.75
                 self.Conditions = ['Crosses', 'Blob', 'SizeMatch', 'Bipartite', 'Pyramids',
                      'Annuli', 'Incomplete', 'Robinson-deSa2013', 
                      'Robinson-deSa2012-E1', 'Robinson-deSa2012-E2', 'Prediction']
                 self.condition = condition
                 self.timeStep = 0.1
                 self.stopTime = 6.0 
                 self.startTime = 0.1
                 self.testOnset = self.stopTime-2.0
                 self.testColorChange=5
                 self.gdAcrossWeight = 0.5
                 self.paramA = 50 # 1 
                 self.paramB = 90 # 90
                 self.paramD = 60 # 60
                 self.boundaryUpperLimit=25
                 # for k in np.arange(0,nOrient):
                 self.F=[]
                 self.orthgonalK1 = 1
                 self.orthgonalK2 = 0
                 self.inputImage = []
                 self.orientedImage = np.ones((self.i_x, self.i_y,3))
                 self.step = 1   # can sub-sample image if bigger than 1 
                 self.orientSize = 0
                 self.LGNwb = []
                 self.gate =[]
                 self.O2 =[]
                 self.sX = 0
                 self.sY = 0
                 self.wb =[]
                 self.w1 = []
                 self.S_wb = []
                 self.S_rg = []
                 self.S_by = []
                 self.LGNkernels()
                 self.createStimulus()
                 self.LGNCells()
                 self.simpleCell()
                 self.complexCell()
                 self.gateComp()
                 self.dipoleComp()
    
    def LGNkernels(self): #  Define LGN kernels and other paramters
        """
        creates Fourier filters used to extract orientation specific image qualities.
        
        Parameters
        -----------------------------------------------------------------------
        alpha/beta/gamma : float value
                         Parameter of Gaussian density function defining the
                         "radius of spreads" (alpha<beta)
        orientationShift : float value
                         Boundaries are shifted relative to image plane
                         (boundaries are between pixels) Default 0.5
        K                : int value
                         Number of orientation polarities 
                         (4 means horizontal and vertical orientations)
           
        Returns
        -----------------------------------------------------------------------
        F                : numpy array (float64)
                         Array containing the Fourier filters images of K-different
                         polarities
        
        """
        C_ = 10 # coefficient (C>E) 18        
        # excitatory condition
        self.C = np.round(Gaussian2D([0,0], C_, self.alpha, 2*np.log(2)),3)
        E_ = .5  # coefficient
        # inhibitory condition
        self.E = np.round(Gaussian2D([0,0], E_, self.beta, 2*np.log(2)),3)
        
        G = Gaussian2D([0+self.orientationShift,0+self.orientationShift], 1, self.gamma, 2)
        self.F = np.zeros((G.shape[0],G.shape[1],4))
        
        # Orientation filters (Difference of Offset Gaussians)
        for k in np.arange(0,self.K):
            m = np.sin((2*np.pi*(k+1))/self.K)
            n = np.cos((2*np.pi*(k+1))/self.K)
            H = Gaussian2D([m+self.orientationShift,n+self.orientationShift],1,self.gamma,2)
            self.F[:,:,k] = G - H
            
            # normalize positive and negative parts
            posF = self.F[:,:,k]
            (posF>90).choose(posF,0)
            posnorm = np.sum(np.sum(posF*posF))
            posF = posF/np.sqrt(posnorm)
            
            negF = self.F[:,:,k]
            (negF>90).choose(negF,0)
            negnorm = np.sum(np.sum(negF*negF))
            negF = negF/np.sqrt(negnorm)
            
            self.F[:,:,k] = posF + negF
            
            # normalize full kernel
            normalizer = np.sum(np.sum(self.F[:,:,k]*self.F[:,:,k] ) )
            self.F[:,:,k] =self.F[:,:,k]/np.sqrt(normalizer)
    
    
    def createStimulus(self):
        """
        Creates the images which will be analysed by the work flow. There are 10 
        conditions which display different types of contour adaptation.
        
        Parameters
        -----------------------------------------------------------------------
        condition        : int value
                         A number which labels the image condition [0-10]
        resultsDirectory : where to save the final imags
        timeStep         : time between images
        timeCount        :point in time
        startTime        :initial time
        stopTime         : end time
        timeOnset        : point at which adapting bars go and C.A. observed
        testColorChange  : 
        midPoint         : centre point of the image
        inputImage       : an image of the condition at a specific time point
        testColorChange  : this is the change, increase or decrease from the 
                         gray background (5 is typical for A&G conditions )
        adaptorSize      : size of adapting edge
        startInputImage  : background onto which the condition is built


        Returns
        -----------------------------------------------------------------------
        C/E              : Gaussian filters Excitatory/Inhibitory        
        rg/by/wb         : individual RGB color channels of output image at specific time        
        
        """
        print "Simulation condition : ", self.Conditions[self.condition]
        
        # Create directory for results
        self.resultsDirectory = os.path.dirname("{0}/{1}".format("Condition", self.Conditions[self.condition]))
        if os.path.exists(self.resultsDirectory)==0:
            os.mkdir(self.resultsDirectory)
        
        if self.condition==7:  # special settings for R&dS 2013 condition
            self.timeStep = 0.153 # seconds
            self.stopTime = 8.12 # seconds -- 6.12 seconds of adaptation
            #stopTime = 2+timeStep # seconds - no adapt condition (actually one presentation) -- 0.153 seconds of adaptation
            self.startTime = self.timeStep
            self.testOnset = self.stopTime-2.0 # seconds
            self.testColorChange = 8  # 8 is good for demonstration
        
        if self.condition==8 or self.condition==9: # special settings for R&dS 2012 conditions
            self.timeStep = 0.16 # seconds
            self.stopTime = 7.0 # seconds -- 5 seconds of adaptation (R&dS 2012 had long initial period and then short re-adapt)
            #stopTime = 2+3*timeStep # seconds -no adapt condition (actually one flicker cycle of adaptation)
            self.startTime = self.timeStep
            self.testOnset = self.stopTime-2.0 # seconds
            self.testColorChange = 8  # 8 is good for demonstration of same size and laself.self.rger size adaptor  
        
        # Initiate time sequence
        self.timeCount=0
        for t in np.arange(self.startTime,self.stopTime,self.timeStep):
            self.timeCount= self.timeCount+1
            
            # change adaptor color with each time step to produce flicker
            self.adaptorColorChange = -self.gray # black
            if np.mod(self.timeCount, 2)== 0:
                self.adaptorColorChange=self.gray
                
               
            if self.condition==0: # Crosses (Adapter size self.by divisible self.by 2 & 6)
                self.adaptorSize=42            
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray  
                for i in np.arange(1,5): # for range 1-4
                    self.midPoint =[0,0]
                    if i==1: # left
                       self.midPoint = [self.i_x/2, self.i_y/4]
                    if i==2: # right
                       self.midPoint = [self.i_x/2, 3*self.i_y/4]
                    if i==3: # top
                       self.midPoint = [1*self.i_x/4, self.i_y/2]
                    if i==4: # bottom
                       self.midPoint = [3*self.i_x/4, self.i_y/2]   
                    self.midPoint[0] = np.round(self.midPoint[0])
                    self.midPoint[1] = np.round(self.midPoint[1])
                    if t < self.testOnset: # draw adaptors   
                        # draw crosses
                        if i==3 or i==4:
                            # vertical
                            startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2:1, self.midPoint[1]-self.adaptorSize/6:self.midPoint[1]+self.adaptorSize/6:1]=self.gray+self.adaptorColorChange
                            # horizontal
                            startinputImage[self.midPoint[0]-self.adaptorSize/6:self.midPoint[0]+self.adaptorSize/6:1, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2:1]=self.gray+self.adaptorColorChange
                            # make outline, self.by cutting middle
                            startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1:1, self.midPoint[1]-self.adaptorSize/6+1:self.midPoint[1]+self.adaptorSize/6-1:1]=self.gray
                            startinputImage[self.midPoint[0]-self.adaptorSize/6+1:self.midPoint[0]+self.adaptorSize/6 -1:1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1:1]=self.gray
                    else: # draw test stimuli
                        self.testColor = self.gray+self.testColorChange
                        if i==1 or i==3:
                            self.testColor=self.gray-self.testColorChange
                        # vertical
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2:1, self.midPoint[1]-self.adaptorSize/6:self.midPoint[1]+self.adaptorSize/6:1]=self.testColor
                        # horizontal
                        startinputImage[self.midPoint[0]-self.adaptorSize/6:self.midPoint[0]+self.adaptorSize/6:1, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2:1]=self.testColor   
                self.startinputImage = startinputImage
            
            if self.condition==1: # Blob 
                self.adaptorSize=42
                self.midPoint=[0,0]
                startinputImage = np.zeros((self.i_x, self.i_y))
                if t< self.testOnset:  # draw adaptors   
                    # right (blurry square)
                    self.midPoint = [self.i_x/2, 3*self.i_y/4]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]= self.adaptorColorChange
                    # blur entire image before putting other elements on
                    temp = np.ones(10)/100;
                    Blur = scipy.ndimage.convolve1d(startinputImage, temp)
                    startinputImage = Blur + self.gray;  
                    # left
                    self.midPoint = [self.i_x/2, self.i_y/4];
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                    startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                else: # draw test stimuli
                   startinputImage =  np.ones((self.i_x, self.i_y))*self.gray;
                   self.midPoint = [self.i_x/2, self.i_y/4];
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                   self.midPoint = [self.i_x/2, 3*self.i_y/4];
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange    
                return startinputImage
            
            if self.condition==2: # SizeMatch 
                self.adaptorSize=42
                self.midPoint=[0,0]
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray;
                if t < self.testOnset:  # draw adaptors   
                   # left
                   self.adaptorSize = 54
                   self.midPoint = [round(2*self.i_x/3), self.i_y/4]
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                   startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                   # right
                   self.adaptorSize = 30
                   self.midPoint = [round(2*self.i_x/3),3*self.i_y/4]
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                   startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                   # top
                   self.adaptorSize = 42
                   self.midPoint = [self.i_x/4,2*self.i_y/4]
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                   startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray  
                else: # draw test stimuli
                    # left
                    self.midPoint = [round(2*self.i_x/3), self.i_y/4]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                    # right
                    self.midPoint = [round(2*self.i_x/3), 3*self.i_y/4]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                    # top
                    self.midPoint = [self.i_x/4, self.i_y/2]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                return startinputImage
            
            if self.condition==3: # Bipartite 
                self.adaptorSize=120
                self.midPoint=[0,0]
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray
                if t < self.testOnset:  # draw adaptors   
                     # center vertical line
                     self.midPoint = [self.i_x/2, self.i_y/2]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]:self.midPoint[1]]=self.gray+self.adaptorColorChange;              
                else:
                     # draw test stimuli
                     # darker self.gray for background
                     startinputImage = (np.ones((self.i_x, self.i_y))*self.gray)-30 
                     # left side
                     self.midPoint = [self.i_x/2, self.i_y/2]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]]=self.gray+self.testColorChange                  
                      # right side
                     self.midPoint = [self.i_x/2, self.i_y/2]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]:self.midPoint[1]+self.adaptorSize/2]=self.gray
                return startinputImage
            
            if self.condition==4: # Pyramids
                self.midPoint=[0,0]
                t=self.testOnset
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray
                if t < self.testOnset:  # draw adaptors
                    for self.adaptorSize in np.arange(62,10,-16):
                         # right 
                         self.midPoint = [self.i_x/2, 3*self.i_y/4]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                         # left
                         self.midPoint = [self.i_x/2, self.i_y/4]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                else: # draw test stimuli
                    startinputImage =  np.ones((self.i_x, self.i_y))*self.gray
                    # draw pyramids
                    count=1
                    for self.adaptorSize in np.arange(62,10,-16):
                        self.midPoint = [self.i_x/2, self.i_y/4]
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray-self.testColorChange*count
                        self.midPoint = [self.i_x/2, 3*self.i_y/4]
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange*count    
                        count=count+1
                    count=count-1
                    # draw comparison centers
                    self.midPoint = [self.i_x/4, self.i_y/4]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray-self.testColorChange*count
                    self.midPoint = [self.i_x/4, 3*self.i_y/4]
                    startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange*count    
                self.startinputImage=startinputImage
            
            if self.condition==5: # Annuli 
                self.midPoint=[0,0]
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray
                if t < self.testOnset:  # draw adaptors  (lines are thicker than for other demos) 
                     # right, top
                     self.midPoint = [self.i_x/4, 3*self.i_y/4]
                     self.adaptorSize=42+2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.adaptorSize= self.adaptorSize-2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # right, bottom
                     self.adaptorSize=18+2
                     self.midPoint = [3*self.i_x/4, 3*self.i_y/4]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.adaptorSize= self.adaptorSize-2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # left, top
                     self.adaptorSize=42+2
                     self.midPoint = [self.i_x/4, self.i_y/4]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.adaptorSize= self.adaptorSize-2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # left, bottom
                     self.midPoint = [3*self.i_x/4, self.i_y/4]
                     self.adaptorSize=18+2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.adaptorSize= self.adaptorSize-2
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray              
                else: # draw test stimuli                
                     # right, top
                     self.adaptorSize=42
                     self.midPoint = [self.i_x/4, 3*self.i_y/4]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                     self.adaptorSize=18
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # right, bottom
                     self.adaptorSize=42
                     self.midPoint = [3*self.i_x/4, 3*self.i_y/4]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                     self.adaptorSize=18
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # left, top
                     self.adaptorSize=42
                     self.midPoint = [self.i_x/4, self.i_y/4]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray-self.testColorChange
                     self.adaptorSize=18
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                     # left, bottom
                     self.midPoint = [3*self.i_x/4, self.i_y/4]
                     self.adaptorSize=42
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray-self.testColorChange
                     self.adaptorSize=18;
                     startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray     
                self.startinputImage=startinputImage
                
            if self.condition==6: # Incomplete 
                self.adaptorSize=84
                self.midPoint=[0,0]
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray
                if t < self.testOnset:  # draw adaptors   
                  self.midPoint = [self.i_x/2, self.i_y/2];
                  startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                  startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                  # blank out right side
                  startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]:self.midPoint[1]+self.adaptorSize/2]=self.gray
                else: # draw test stimuli
                   startinputImage =  np.ones((self.i_x, self.i_y))*self.gray
                   self.midPoint = [self.i_x/2, self.i_y/2]
                   startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange   
                self.startinputImage = startinputImage
                
            if self.condition==7: # Robinson-deSa 2013
                self.adaptorSize=42
                self.midPoint=[0,0]
                startinputImage = np.zeros((self.i_x, self.i_y))
                if t < self.testOnset:  # draw adaptors   
                     startinputImage[:,:]= self.gray + self.adaptorColorChange
                     # right 
                     self.midPoint = [self.i_x/2, 3*self.i_y/4]
                     startinputImage[self.midPoint(1)-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint(2)-self.adaptorSize/2:self.midPoint(2)+self.adaptorSize/2]= self.gray
                     # left
                     self.midPoint = [self.i_x/2, self.i_y/4]
                     startinputImage[self.midPoint(1)-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint(2)-self.adaptorSize/2:self.midPoint(2)+self.adaptorSize/2]=self.gray
                else: # draw test stimuli
                      # equal-sized test stimulus
                      startinputImage =  np.ones((self.i_x, self.i_y))*self.gray;
                      self.midPoint = [self.i_x/2, self.i_y/4];
                      startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.testColorChange
                      # small test stimulus
                      self.midPoint = [self.i_x/2, 3*self.i_y/4];
                      startinputImage[self.midPoint[0]-round(self.adaptorSize/4):self.midPoint[0]+round(self.adaptorSize/4), self.midPoint[1]-round(self.adaptorSize/4):self.midPoint[1]+round(self.adaptorSize/4)]=self.gray+self.testColorChange    
                return startinputImage
            if self.condition==8: # Robinson-deSa2012-E1 (stimulus sizes multiply degrees self.by 10 for pixels)
                # (self.adaptorSize (100,40,20),self.testSize (20,100))
                self.adaptorSize=100
                self.testSize=100
                self.midPoint=[0,0]
                startinputImage = np.zeros((self.i_x, self.i_y))
                if t < self.testOnset:  # draw adaptors   
                     startinputImage[:,:]= self.gray
                      # centered 
                     self.midPoint = [self.i_x/2, self.i_y/2]
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]= self.gray + self.adaptorColorChange
                else: # draw test stimulus
                      startinputImage =  np.ones(self.i_x, self.i_y)*self.gray
                      # left
                      self.midPoint = [self.i_x/2, self.i_y/2]
                      startinputImage[self.midPoint[0]-self.testSize/2:self.midPoint[0]+self.testSize/2, self.midPoint[1]-self.testSize/2:self.midPoint[1]+self.testSize/2]=self.gray+self.testColorChange  
                return startinputImage
            if self.condition==9: # Robinson-deSa2012-E2 (stimulus sizes multiply degrees self.by 10 for pixels)
                # (self.adaptorSize, middleself.adaptorSize, innerself.adaptorSize,self.testSize)
                self.adaptorSize=100
                self.middleadaptorSize=100
                self.inneradaptorSize=0
                self.testSize=100
                self.midPoint=[0,0]
                startinputImage = np.zeros((self.i_x, self.i_y))
                if t < self.testOnset:  # draw adaptors   
                     startinputImage[:,:]= self.gray
                     # centered 
                     self.midPoint = [self.i_x/2, self.i_y/2]
                     # outer edge
                     # upper left
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0], self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]]= self.gray + self.adaptorColorChange
                     # bottom left
                     startinputImage[self.midPoint[0]:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]]= self.gray - self.adaptorColorChange
                     # upper right
                     startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0], self.midPoint[1]:self.midPoint[1]+self.adaptorSize/2]= self.gray - self.adaptorColorChange
                     # bottom right
                     startinputImage[self.midPoint[0]:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]:self.midPoint[1]+self.adaptorSize/2]= self.gray + self.adaptorColorChange
                     # middle edge
                     # upper left
                     startinputImage[self.midPoint[0]-self.middleadaptorSize/2:self.midPoint[0], self.midPoint[1]-self.middleadaptorSize/2:self.midPoint[1]]= self.gray - self.adaptorColorChange
                     # bottom left
                     startinputImage[self.midPoint[0]:self.midPoint[0]+self.middleadaptorSize/2, self.midPoint[1]-self.middleadaptorSize/2:self.midPoint[1]]= self.gray + self.adaptorColorChange
                     # upper right
                     startinputImage[self.midPoint[0]-self.middleadaptorSize/2:self.midPoint[0], self.midPoint[1]:self.midPoint[1]+self.middleadaptorSize/2]= self.gray + self.adaptorColorChange
                     # bottom right
                     startinputImage[self.midPoint[0]:self.midPoint[0]+self.middleadaptorSize/2, self.midPoint[1]:self.midPoint[1]+self.middleadaptorSize/2]= self.gray - self.adaptorColorChange
                     # self.gray interior
                     if self.inneradaptorSize>0:
                         startinputImage[self.midPoint[0]-self.inneradaptorSize/2:self.midPoint[0]+self.inneradaptorSize/2, self.midPoint[1]-self.inneradaptorSize/2:self.midPoint[1]+self.inneradaptorSize/2]= self.gray
                else: # draw test stimulus
                      startinputImage =  np.ones((self.i_x, self.i_y))*self.gray
                      # left
                      self.midPoint = [self.i_x/2, self.i_y/2]
                      startinputImage[self.midPoint[0]-self.testSize/2:self.midPoint[0]+self.testSize/2, self.midPoint[1]-self.testSize/2:self.midPoint[1]+self.testSize/2]=self.gray+self.testColorChange 
                return startinputImage

            if self.condition==10: # Prediction 
                self.testSize=50
                self.midPoint=[0,0]
                startinputImage = np.ones((self.i_x, self.i_y))*self.gray
                if t < self.testOnset:  # draw adaptors 
                    # illusory contour on left
                    for self.adaptorSize in np.arange(32,5,-8):
                        # topright 
                        self.midPoint = [3*self.i_x/8, self.i_y/2-self.testSize/2 - self.testSize];
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                        startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                        # leftright
                        self.midPoint = [3*self.i_x/8, self.i_y/2+self.testSize/2- self.testSize];
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                        startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
            
                        # bottomright 
                        self.midPoint = [3*self.i_x/8+self.testSize, self.i_y/2-self.testSize/2- self.testSize]
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                        startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                        # bottomleft
                        self.midPoint = [3*self.i_x/8+self.testSize, self.i_y/2+self.testSize/2- self.testSize]
                        startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                        startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray  
                    # middle self.gray  
                    tempinputImage = np.ones(self.i_x, self.i_y)*self.gray
                    # copy and paste from temp to startinput
                    self.midPoint = [self.i_x/2, self.i_y/2- self.testSize]
                    startinputImage[self.midPoint[0]-self.testSize/2+1:self.midPoint[0]+self.testSize/2 -1, self.midPoint[1]-self.testSize/2+1:self.midPoint[1]+self.testSize/2-1]=tempinputImage[self.midPoint[0]-self.testSize/2+1:self.midPoint[0]+self.testSize/2 -1, self.midPoint[1]-self.testSize/2+1:self.midPoint[1]+self.testSize/2-1]                   
            
                    # drawn contour on right
                    for self.adaptorSize in np.arange(32,5,-8):
                        # topright 
                         self.midPoint = [3*self.i_x/8, self.i_y/2-self.testSize/2 + self.testSize]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                         # leftright
                         self.midPoint = [3*self.i_x/8, self.i_y/2+self.testSize/2+ self.testSize]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
             
                         # bottomright 
                         self.midPoint = [3*self.i_x/8+self.testSize, self.i_y/2-self.testSize/2+ self.testSize]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                         # bottomleft
                         self.midPoint = [3*self.i_x/8+self.testSize, self.i_y/2+self.testSize/2+ self.testSize]
                         startinputImage[self.midPoint[0]-self.adaptorSize/2:self.midPoint[0]+self.adaptorSize/2, self.midPoint[1]-self.adaptorSize/2:self.midPoint[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                         startinputImage[self.midPoint[0]-self.adaptorSize/2+1:self.midPoint[0]+self.adaptorSize/2 -1, self.midPoint[1]-self.adaptorSize/2+1:self.midPoint[1]+self.adaptorSize/2-1]=self.gray
                         
                    # draw line 
                    self.midPoint = [self.i_x/2, self.i_y/2 + self.testSize]
                    startinputImage[self.midPoint[0]-self.testSize/2+1:self.midPoint[0]+self.testSize/2 , self.midPoint[1]-self.testSize/2+1:self.midPoint[1]+self.testSize/2]=self.gray+self.adaptorColorChange
                    startinputImage[self.midPoint[0]-self.testSize/2+2:self.midPoint[0]+self.testSize/2 -1 , self.midPoint[1]-self.testSize/2+2:self.midPoint[1]+self.testSize/2-1]=self.gray
                   
                else: # draw test stimuli
                    startinputImage =  np.ones((self.i_x, self.i_y))*self.gray
                    #left self.gray
                    self.midPoint = [self.i_x/2, self.i_y/2 - self.testSize]
                    startinputImage[self.midPoint[0]-self.testSize/2+1:self.midPoint[0]+self.testSize/2 -1, self.midPoint[1]-self.testSize/2+1:self.midPoint[1]+self.testSize/2-1]=self.gray+self.testColorChange
                    #right self.gray
                    self.midPoint = [self.i_x/2, self.i_y/2 + self.testSize]
                    startinputImage[self.midPoint[0]-self.testSize/2+1:self.midPoint[0]+self.testSize/2 -1, self.midPoint[1]-self.testSize/2+1:self.midPoint[1]+self.testSize/2-1]=self.gray+self.testColorChange 
                return startinputImage
    
            # fixation markers
            self.startinputImage[self.i_x/2-1,self.i_y/2-1]=255
            self.startinputImage[self.i_x/2+1,self.i_y/2+1]=255
            self.startinputImage[self.i_x/2-1,self.i_y/2+1]=0
            self.startinputImage[self.i_x/2+1,self.i_y/2-1]=0
            
            """
            if makeAnimatedGifs==1:
                filename = "{0}/{1}".format(resultsDirectory,'Stimulus.gif')
                if timeCount==1:
                    cv2.imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'loopcount',inf)
                else:
                    cv2.imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'writemode','append')
            """
            
            # Convert self.rgb input image to red-green, blue-yellow, white-black coordinates
            self.inputImage = np.zeros((self.i_x, self.i_y, 3))
            self.inputImage[:,:,0] = startinputImage
            self.inputImage[:,:,1] = startinputImage
            self.inputImage[:,:,2] = startinputImage
            [self.rg, self.by, self.wb] = ConvertRGBtoOpponentColor(self.inputImage, self.gray)
        
            
    def LGNCells(self):    
        """   
        The portion of the code represents theoretial processing methods 
        employed by LGN cells. The white-black channel is convolved with
        gaussian filters, shunted and then padded.
        
        [center-surround processing, but with current parameters it actually
        does very little.]
        
        Parameters
        -----------------------------------------------------------------------
        wb           : white-black color channel of image
        wb2          : padded white-black color channel of image
        C/E          : Excititory/Inhibitory Gaussian Filters
        PaddingColor : Grey scale value of padding Default 0
        paramA/B/D   : Shunting parameters
        
        Returns
        -----------------------------------------------------------------------
        LGNwb        - white-black LGN cell signal
        by/wb/rg     - padded color channels
        
        """
         # padding
        self.PaddingSize = math.floor(np.max(self.wb.shape)/2) 
        PaddingColor = self.wb[0,0]
        self.wb2 = im_padding(self.wb, self.PaddingSize, PaddingColor)
        
        # convolution - replicates MATLABS conv2 function 
        OnOff_Excite =  conv2(self.wb2, self.C, mode='same')
        OnOff_Inhibit = conv2(self.wb2, self.E, mode='same')
        OffOn_Excite =  conv2(self.wb2, self.E, mode='same')
        OffOn_Inhibit = conv2(self.wb2, self.C, mode='same')
        
        # Faster "Raw" Version of Convolution
        #    OnOff_Excite =  np.fft.irfft2(np.fft.rfft2(self.wb2) * np.fft.rfft2(C, self.wb2.shape))
        #    OnOff_Inhibit = np.fft.irfft2(np.fft.rfft2(self.wb2) * np.fft.rfft2(E, self.wb2.shape))
        #    OffOn_Excite =  np.fft.irfft2(np.fft.rfft2(self.wb2) * np.fft.rfft2(E, self.wb2.shape))
        #    OffOn_Inhibit = np.fft.irfft2(np.fft.rfft2(self.wb2) * np.fft.rfft2(C, self.wb2.shape))
        
        # shunting
        x_OnOff = (self.paramB*OnOff_Excite-self.paramD*OnOff_Inhibit)/(self.paramA+(OnOff_Excite+OnOff_Inhibit))
        x_OffOn = (self.paramB*OffOn_Excite-self.paramD*OffOn_Inhibit)/(self.paramA+(OffOn_Excite+OffOn_Inhibit))
        
        # cutting negative values
        x_pos = x_OnOff - x_OffOn
        x_pos[x_pos<0] = 0
        x_neg = x_OffOn - x_OnOff
        x_neg[x_neg<0] = 0
        self.LGNwb = x_pos - x_neg
        
        # pad planes for all color channels for later use
        PaddingColor = self.wb[0,0]
        self.wb = im_padding(self.wb, self.PaddingSize, PaddingColor)
        PaddingColor = self.rg[0,0]
        self.rg = im_padding(self.rg, self.PaddingSize, PaddingColor)
        PaddingColor = self.by[0,0]
        self.by = im_padding(self.by, self.PaddingSize, PaddingColor)
                    
        
    def simpleCell(self):
        """
        Model of simple cell  processing. LGN cell signal is convolved with
        orientation specific gaussian filters. Images are then cropped and set 
        to be < 0.
        
        Parameters
        -----------------------------------------------------------------------
        LGNwb       : white-black signal from LGN cells
        F           : orientation specific gaussian filters
        PaddingSize : size of padding
        K           : number of polarities
        
        
        Returns
        -----------------------------------------------------------------------
        y           : simple cell image signal (sX,sX,K) 
        
        """
        # Orientations are only based on inputs from the white-black color channel
        
        #    y_pos_1 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNself.wb) * np.fft.rfft2(F[:,:,0], LGNself.wb.shape)))
        #    y_pos_2 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNself.wb) * np.fft.rfft2(F[:,:,1], LGNself.wb.shape)))
        #    y_pos_3 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNself.wb) * np.fft.rfft2(F[:,:,2], LGNself.wb.shape)))
        #    y_pos_4 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNself.wb) * np.fft.rfft2(F[:,:,3], LGNself.wb.shape)))
        
        y_pos_1_ = conv2(self.LGNwb, self.F[:,:,0], mode='same') # loses imaginary components?
        y_pos_2_ = conv2(self.LGNwb, self.F[:,:,1], mode='same')
        y_pos_3_ =  conv2(self.LGNwb, self.F[:,:,2], mode='same')
        y_pos_4_ = conv2(self.LGNwb, self.F[:,:,3], mode='same')    
        
        y_crop_1 = im_cropping(y_pos_1_, self.PaddingSize)
        y_crop_2 = im_cropping(y_pos_2_, self.PaddingSize) 
        y_crop_3 = im_cropping(y_pos_3_, self.PaddingSize) 
        y_crop_4 = im_cropping(y_pos_4_, self.PaddingSize) 
        
        y_crop_1[y_crop_1<0] = 0
        y_crop_2[y_crop_2<0] = 0
        y_crop_3[y_crop_3<0] = 0 
        y_crop_4[y_crop_4<0] = 0
        
        y_crop=np.zeros((y_crop_1.shape[0],y_crop_1.shape[1],self.K))
        
        y_crop[:,:,0] = y_crop_1
        y_crop[:,:,1] = y_crop_2
        y_crop[:,:,2] = y_crop_3
        y_crop[:,:,3] = y_crop_4
        
        self.y = y_crop
        
    def complexCell(self):
        """
        Model of complex cell processing. 
        
        Parameters
        -----------------------------------------------------------------------
        y                  : filtered LGN cell signals from simple cells
        K                  : number of polarities
        nOrient            : number of orientations
        boundaryUpperLimit : upper limit of boundary activity
        inI                : tonic input
        
        
        Returns
        -----------------------------------------------------------------------
        w1                 : complex cell output image signal
        
        """
        # pool across contrast polarity [ 0 -> (0 + 2) / 1 -> (1 + 2) ]
        z1 = np.zeros((self.y.shape[0], self.y.shape[1], self.nOrient))
        for k in np.arange(0,self.K/2):
            z1[:,:,k] = self.y[:,:,k] + self.y[:,:,k+self.K/2]
        
        # set upper limit for boundary activity
        z1[z1>self.boundaryUpperLimit] = self.boundaryUpperLimit
       
        # Add tonic input, inI, to boundaries        
        self.w1 = np.zeros((self.y.shape[0], self.y.shape[1], self.nOrient))
        for k in np.arange(0,self.nOrient):
            self.w1[:,:,k] = self.inI  + z1[:,:,k]
        return self.w1
        
    def gateComp(self):
        """
        Orientation specific habituating transmitter gates
        
        Parameters
        -----------------------------------------------------------------------
        A/B/C gate  - gate parameters
        w1          - output of complex cell
        inI         - tonic input
        Rhogate     - 
        timeStep    - time between images (seconds)
        
        Returns
        -----------------------------------------------------------------------
        gate        -  the value of the response of the O.S.H.T gates
        
        """
        gate = np.zeros(self.w1.shape)
        # initialize gate on first time step
        t = self.startTime
        if t ==self.startTime :
            gate = self.Agate/(self.Bgate + self.Cgate*self.inI) * np.ones(self.w1.shape)
        else:
            gate = np.ones(self.w1.shape)
        
        # identify equilibrium solution to gate
        gate_equil = self.Agate/(self.Bgate + self.Cgate* self.w1)
        
        # solve gate for current time
        self.gate = gate_equil + (gate - gate_equil)* np.exp(-self.Rhogate*(self.Bgate+self.Cgate*self.w1)*self.timeStep)
    
    
    def dipoleComp(self):
        """
        Cross orientation dipole competition
        
        Parameters
        -----------------------------------------------------------------------
        gate            - 
        wi              - 
        gdAcrossWeight  - strength of inhibition between orthogonal orientations
        orthogonalK1/K2 - 
        
        
        Returns
        -----------------------------------------------------------------------
        O2              - Signal after cross orientation dipole competition
        """                        
        # Gated signal minus inhibited gated signal       
        v_1 = self.gate[:,:,0]*self.w1[:,:,0] - self.gdAcrossWeight*self.gate[:,:,self.orthgonalK1]*self.w1[:,:,self.orthgonalK1] 
        v_2 = self.gate[:,:,1]*self.w1[:,:,1] - self.gdAcrossWeight*self.gate[:,:,self.orthgonalK2]*self.w1[:,:,self.orthgonalK2] 
        """ Error involved here in v_2 unknown origin"""
        
        # half-wave rectify
        v_1[v_1<0] = 0  
        v_2[v_2<0] = 0
        
        O1=np.zeros((v_1.shape[1],v_1.shape[0],self.nOrient))
        O1[:,:,0] = v_1
        O1[:,:,1] = v_2
        
        # soft threshold for boundaries
        self.O2 = O1-self.bThresh
        self.O2[self.O2<0] = 0
    
        
    def fillingFIDO(self):
        """
        FIlling-in DOmains [FIDOs] - regions of connected boundary points are
        located and then filled in. Boundaries are shifted in a grid plane 
        relative to color/brightness signals
        
        Parameters
        -----------------------------------------------------------------------
        O2                : gated signal
        BndThr            : Boundary threshold
        BndSig            : Boundary signal
        nOrient           : number of filter orientations
        stimarea_x/y      : area of stimulation
        shift             : up, down, left, right
        shiftOrientation  : orientation of flow 
        Bshift            : Boundaries that block flow for the corresponding shift 
                
        Returns
        -----------------------------------------------------------------------
        S_wb/rg/by        : filled in images of respective color channels
        
        """
        # Most of this code is an algorithmic way of identifying distinct FIDOs
        BndSig = np.sum(self.O2[:,:,:],2) 
        thint = self.O2.shape
        BndThr = 0.0
        BndSig = 100*(BndSig - BndThr)
        BndSig[BndSig < 0] = 0
        
        boundaryOrientation = np.zeros((1,self.nOrient))
        for i in np.arange(0,self.nOrient):
            boundaryOrientation[0,i] = -np.pi/2 +(i+1)*np.pi/(self.nOrient)
        
        self.sX = np.size(BndSig, 0)
        self.sy = np.size(BndSig, 1)
        
        stimarea_x = np.arange(1,np.size(BndSig, 0)-1) 
        stimarea_y = np.arange(1,np.size(BndSig, 1)-1)
        
        # Setting up boundary structures
        P=np.zeros((self.sX,self.sy,4))
        for i in np.arange(0,4):
            dummy = np.ones((self.sX, self.sy))
            p1 = stimarea_x + self.Bshift1[i,0]
            q1 = stimarea_y + self.Bshift1[i,1]
            p2 = stimarea_x + self.Bshift2[i,0]
            q2 = stimarea_y + self.Bshift2[i,1]
            
            currentBoundary = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            currentBoundary1 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            currentBoundary2 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            
            # for both orientations at each polarity
            a1=np.abs(np.sin(self.shiftOrientation[i] - boundaryOrientation[0,0]))
            currentBoundary1[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = a1*(self.O2[p1[0]:p1[-1]:1,q1[0]:q1[-1]:1,0] + self.O2[p2[0]:p2[-1]:1,q2[0]:q2[-1]:1,0] )
            
            a2=np.abs(np.sin(self.shiftOrientation[i] - boundaryOrientation[0,1]))
            currentBoundary2[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = a2*(self.O2[p1[0]:p1[-1]:1,q1[0]:q1[-1]:1,1] + self.O2[p2[0]:p2[-1]:1,q2[0]:q2[-1]:1,1] )
            
            currentBoundary=currentBoundary1+currentBoundary2
            a = currentBoundary
            a[a>0] = 1
            a1=dummy[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1]
            a2=    a[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1]
            
            P[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1,i] =   a1- a2
        
        
        # find FIDOs and average within them
        FIDO = np.zeros((self.sX, self.sy))
        
        # unique number for each cell in the FIDO
        for i in np.arange(0,self.sX):
            for j in np.arange(0,self.sy):
                FIDO[i,j] = i+ j*thint[0]
                
        # Grow each FIDO so end up with distinct domains with a common assigned number
        oldFIDO = np.zeros((self.sX, self.sy))
        while np.array_equal(oldFIDO, FIDO) ==0:
            oldFIDO = FIDO;
            for i in np.arange(0,4):
                p = stimarea_x + self.shift[i,0] ############ -1 SPLINT ARRAy SIZE ERROR
                q = stimarea_y + self.shift[i,1] ############ -1 SPLINT ARRAy SIZE ERROR
                FIDO[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = np.maximum(FIDO[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1], FIDO[p[0]:p[-1]:1,q[0]:q[-1]:1]*P[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1,i])
        
        # input is color signals
        self.wbColor = im_cropping(self.wb, self.PaddingSize)
        self.rgColor = im_cropping(self.rg, self.PaddingSize)
        self.byColor = im_cropping(self.by, self.PaddingSize)
        
        # Filling-in values for white-black, red-green, and blue-yellow
        self.S_wb = np.zeros((self.sX, self.sY))
        self.S_rg = np.zeros((self.sX, self.sY))
        self.S_by = np.zeros((self.sX, self.sY)) 
        
        # Compute average color for unique FIDOs
        uniqueFIDOs = np.unique(FIDO)
        numFIDOs = uniqueFIDOs.shape  
        dummyFIDO = np.ones((self.sX,self.sy))
        # Number of pixels in this FIDO
        for i in np.arange(0,numFIDOs[0]):
            Lookup=FIDO==uniqueFIDOs[i]
            FIDOsize = np.sum(np.sum(dummyFIDO[Lookup]))
            # Get average of color signals for this FIDO
            self.S_wb[Lookup] = np.sum(self.wbColor[Lookup])/FIDOsize
            self.S_rg[Lookup] = np.sum(self.rgColor[Lookup])/FIDOsize
            self.S_by[Lookup] = np.sum(self.byColor[Lookup])/FIDOsize
            
    def saveImages(self): 
        """
        Save image files of network behavior
        
        Parameters
        -----------------------------------------------------------------------
        orientedImage    :
        O2               :
        
        Returns
        -----------------------------------------------------------------------
        thing     - final triad of images as specific time frame
        
        """
        
        # transform boundary values into intensity signals for viewing image 
        for i in np.arange(0,self.i_x,self.step): # -step SPLINT
            for j in np.arange(0,self.i_y,self.step): # -step SPLINT
                # if vertical edge at this pixel, color it green
                if self.O2[i,j,1] >0:
                    ratio = self.O2[i,j,1]/80
                    if ratio<0.2:
                        ratio = 0.2
                    self.orientedImage[i,j,2] = 1-ratio # reduce blue -k SPLINT
                    self.orientedImage[i,j,0] = 1-ratio # reduce red  -k SPLINT
        
                # if horizontal edge at this pixel, color it blue
                if self.O2[i,j,0] >0:
                    self.ratio = self.O2[i,j,0]/80
                    if self.ratio<0.2:
                        self.ratio = 0.2
                    self.orientedImage[i, j, 1] = 1-self.ratio # reduce green
                    self.orientedImage[i, j, 0] = 1-self.ratio # reduce red
          
        """
        if makeAnimatedGifs==1:
            [imind,cm] = self.self.rgb2ind(orientedImage,256)
        
            filename = sprintf('%s/Boundaries.gif', resultsDirectory);
        
            if timeCount==1:
                imwrite(imind,cm,filename,'gif','DelayTime',timeStep,'loopcount',inf);
            else:
                imwrite(imind, cm,filename,'gif','DelayTime',timeStep,'writemode','append');
          
        """
        
        # Convert values in the color FIDOs to something that can be presented in an image
        S_rgmax = np.max(np.max(np.abs(self.S_rg[:,:])))
        S_bymax = np.max(np.max(np.abs(self.S_by[:,:])))
        S_wbmax = np.max(np.max(np.abs(self.S_wb[:,:])))
        S_max1 = np.maximum(S_rgmax, S_bymax)
        S_max = np.maximum(S_max1, S_wbmax)
        
        # Convert FIDO values to self.rgb values (relative to self.gray and maximum FIDO value)
        S_rgb = self.ConvertOpponentColortorgB(self.S_rg[:,:], self.S_by[:,:], self.S_wb[:,:], self.gray, S_max)
        # scale to 0 to 255 self.rgb values
        temp = 255.0* (S_rgb[:,:,0]/np.max(np.max(np.max(S_rgb))))
        
        """
        if makeAnimatedGifs==1:
            filename = "{0}/{1}".format("/Filledin.gif", resultsDirectory)
        
            if timeCount==1:
                imwrite(temp,filename,'gif','DelayTime',timeStep,'loopcount',inf)
            else:
                imwrite(temp,filename,'gif','DelayTime',timeStep,'writemode','append')
        """
        # Make image of input, boundaries, and filled-in values to save as a png file
        thing = np.ones((self.i_x, 3*self.i_y, 3))
        
        # Input image on left (Participant Image)
        thing[:,0:self.i_y,:]=self.inputImage/255
        
        # Filled-in values on right (Computer Image)
        thing[:,2*self.i_y:3*self.i_y,0]=temp/255 
        thing[:,2*self.i_y:3*self.i_y,1]=temp/255  
        thing[:,2*self.i_y:3*self.i_y,2]=temp/255
        
        # Boundaries in center (Boundary Image)
        thing[:,self.i_y:2*self.i_y,:]=self.orientedImage # +1 removed from y start SPLINT
        
        # Write individual frame files (with leading zero if less than 10)
        if self.timeCount>=10:
            filename = "{0}/{1}{2}{3}".format(self.resultsDirectory, 'All',self.timeCount,'.png')
        else:
            filename = "{0}/{1}{2}{3}".format(self.resultsDirectory,'All0',self.timeCount,'.png')
        
        #Same image to file
        scipy.misc.imsave(filename, thing)
        
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, figsize=(6,10))
        ax1.imshow(thing[:,:,0])
        ax2.imshow(thing[:,:,1])
        ax3.imshow(thing[:,:,2])
        plt.show()




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

def Gaussian2D(GCenter, Gamp, Ggamma,Gconst):
    new_theta = np.sqrt(Gconst**-1)*Ggamma
    if new_theta < .4:
        print('kernel is too small!')
    SizeHalf = np.int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    part1=(x+GCenter[0])**2+(y+GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)    
    return GKernel
    
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
