# -*- coding: utf-8 -*-
"""
Translated Python Version of Gregory Francis' contour adaptation FACADE model (2015)

-> Extension of Francis and Kim model (2012), where full model description is located.
--> Extension of FACADE (Form And Color and DEpth) model by Grossbery

Basic usage in CANNEM class, upon code running use below command line script to 
convert produced images into a gif.

From the commnand line, go to the directory with the images and type
convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif

To see a overall workflow, go to the evalute function and view the steps inwhich
the model takes to produce the output i.e. calling each of the specific modules.
"""
import numpy as np
import math
import os
import scipy
from scipy.ndimage.filters import convolve
from scipy.ndimage.measurements import mean as labeled_mean
import whitesillusion as wi
    
class CANNEM(object):

    """ 
    Contour Adaptation Neural NEtwork Model- CANNEM
    
    Network flow of contour adaptation model through various stimuli conditions
    
    Basic usage:
    >>> cd "containing Folder"
    >>> import structCAN
    >>> mst=structCAN.CANNEM() # create instance of CANNEM
    >>> mst.evaluate(0)        # run model for stimulus '0'
    
    Note:
    If choosing White's Illusion, i.e. stimuli 11, you have the option of specifying
    orientation of adapters and height of test patches
    
    patch_h=1   ,direction='v'    # Default vertical e.g.
    patch_h=0.25,direction='h'    # Square  square   e.g.
    
    >>> mst.evaluate(11,patch_h=0.25,direction='h')
    
    Parameters
    ----------
    makeAnimatedGifs : int, optional
            binary gif making decision [currently not implemented]
    gray : int
            gray value (127)
    i_x/i_y : int               
            image size   
    K : int                      -
            number of polarities (4 means horizontal and vertical orientations)
    inI : int    
            tonic input
    Conditions : list
            list of condition names
    condition : int
            value of condition
    timeStep : int
            time between images
    stopTime : int
            end time of images
    
    
    Returns
    -----------
    thing : ndarray
            Triad of images representing [Input,Boundaries,Output]
            
    References
    -----------
    [1] Francis, G., & Kim. J. (2012). Simulations of induced visual scene fading with boundary offset and filling-in.
    Vision Research, 62, 181–191
    [2] http://stackoverflow.com/questions/33612568/speeding-up-for-loop-in-image-analysis-when-iterations-are-up-to-40-000
    
    Author
    ----------
    William Baker Morrison
    
    """

    def __init__(self,
                 gray= 127,                         
                 i_x=200, 
                 i_y=200,
                 inI = 5,
                 K = 4 , 
                 Conditions = ['Crosses', 'Blob', 'SizeMatch', 'Bipartite', 'Pyramids',
                      'Annuli', 'Incomplete', 'Robinson-deSa2013', 
                      'Robinson-deSa2012-E1', 'Robinson-deSa2012-E2', 'Prediction','Whites-Illusion'],
                 timeStep = 0.1, # Set timing Parameters (seconds)
                 stopTime = 6, # seconds 6 for A&G conditions - makes for 4 second adaptation
                 testOnset = 6 - 2,
                 ):
                     self.startTime = timeStep
                     self.testOnset = stopTime - 2
                     self.nOrient = K/2
                     self.gray=gray
                     self.i_x = i_x
                     self.i_y=i_y
                     self.inI=inI
                     self.K=K
                     self.Conditions=Conditions
                     self.timeStep=timeStep
                     self.stopTime=stopTime
                     self.testOnset=testOnset
                     # Initiate time sequence 
                     self.timeCount=0
                     
                     
    def evaluate(self,condition,patch_h=1,direction='v'):
        """
        - Kernels are made
        - Simulation begins in 'time' for loop
        - Each step described in more detail within themselves        
        """
        self.patch_h = patch_h
        self.direction = direction
        self.LGNkernels(2*np.log(2), 10, .5, .5, 2, 1.75, 0.5)
        print "Simulation condition : ", self.Conditions[condition]
        for time in np.arange(self.startTime, self.stopTime+self.timeStep, self.timeStep):
            self.timeCount= self.timeCount+1
            self.createStimulus(time,5,condition)
            self.LGNcells()
            self.simpleCell()
            self.complexCell(25)
            self.gateComp(time, 20.0, 1.0, 1.0, 0.007)
            self.dipoleComp()
            self.fillingin()
            self.saveImages(condition)
                         
    def LGNkernels(self,Gconst, C_,E_, alpha,beta,gamma,orientationShift):
        """
        Usage: 
        >>> LGNkernels(self, 2*np.log(2), 10, .5, .5, 2, 1.75, 0.5)
    
        Returns kernels of  lateral geniculate nucleus. The LGN receives input 
        from the retina, as well as non-retinal inputs (excitatory, inhibitory, 
        or modulatory). The axons that leave the LGN go to V1 visual cortex.
        
        A representation of the processing of LGN cells as a single system.
        
        Parameters
        ----------
        Gconst : int
                Parameter of gaussian probability function
        C_/E_ : int
                coefficient (C>E) 18
        alpha/beta/gamma : float
                radius of spreads (a<b) .5
        orientationShift : float
                boundaries (inter-pixel) are shifted relative to image plane
        
        Returns
        ----------
        C/E : array_like
                Gaussian pulses (excitatory/inhibitory)
        F : array_like
                Oriented difference of Gaussian filter (ODOG)
           
        """
        # mathematical avoidance of floating point error
        Gconst= floatingpointtointeger(4,Gconst)
        
        # excitatory 2D Gaussian 
        self.C = Gaussian2D([0,0], C_, alpha, Gconst)
        
        # inhibitory 2D Gaussian
        self.E = Gaussian2D([0,0], E_, beta , Gconst)
        
        # G and H form the kernels which are used to create an ODOG filter
        G = Gaussian2D([0+orientationShift,0+orientationShift], 1, gamma, 2)
        F = np.zeros((G.shape[0],G.shape[1],4))
        self.F=F
        # Orientation filters (Difference of Offset Gaussians)
        for k in np.arange(0,self.K):
            m = np.sin((2*np.pi*(k+1))/self.K)
            n = np.cos((2*np.pi*(k+1))/self.K)
            H = Gaussian2D([m+orientationShift,n+orientationShift],1,gamma,2)
            
            # minus kernels to create ODOG filter for each polarity
            F[:,:,k] = G - H
            
            # normalize positive and negative parts
            posF = F[:,:,k]
            (posF>90).choose(posF,0)
            posnorm = np.sum(np.sum(posF*posF))
            posF = posF/np.sqrt(posnorm)
            
            negF = F[:,:,k]
            (negF>90).choose(negF,0)
            negnorm = np.sum(np.sum(negF*negF))
            negF = negF/np.sqrt(negnorm)
            
            # combine positive and negative parts
            F[:,:,k] = posF + negF
            
            # normalize full kernel
            normalizer = np.sum(np.sum( F[:,:,k]*F[:,:,k] ) )
            self.F[:,:,k] = F[:,:,k]/np.sqrt(normalizer)
    
    def createStimulus(self,time,testColorChange,condition):
        """
        Usage: 
        >>> createStimulus(self,time,5,0)
        

        Holds instructions for how various input stimuli are to be built, as 
        well as their counterpart contours for adaptation.
        
        
        Parameters
        ----------        
        condition : int
                Integer labels are: 'Crosses', 'Blob', 'SizeMatch', 'Bipartite', 
                'Pyramids','Annuli', 'Incomplete', 'Robinson-deSa2013', 
                'Robinson-deSa2012-E1', 'Robinson-deSa2012-E2', 'Prediction' and
                'Whites_Illusion'
        time : float
                current time of simulation to initiate difference stimuli to be 
                produced
        testColorChange : int
                Adaptor color change 5 is typical for A&G conditions 
                (this is the change, increase or decrease, from the gray background)
        
        Returns
        ----------
        rg, by, wb : array_like
                Color channels of stimulus base image
        inputImage : array_like
                Final image shown to participant

        """        
                     
        # Special timing settings for condition 7 & 8 
        if condition==7:  # special settings for R&dS 2013 condition
            self.timeStep = 0.153 # seconds
            self.stopTime = 8.12 # seconds -- 6.12 seconds of adaptation
            #stopTime = 2+timeStep # seconds - no adapt condition (actuallself.y one presentation) -- 0.153 seconds of adaptation
            self.startTime = self.timeStep
            self.testOnset = self.stopTime-2.0 # seconds
            testColorChange = 8  # 8 is good for demonstration
        
        if condition==8 or condition==9: # special settings for R&dS 2012 conditions
            self.timeStep = 0.16 # seconds
            self.stopTime = 7.0 # seconds -- 5 seconds of adaptation (R&dS 2012 had long initial period and then short re-adapt)
            #stopTime = 2+3*timeStep # seconds -no adapt condition (actuallself.y one flicker cself.ycle of adaptation)
            self.startTime = self.timeStep
            self.testOnset = self.stopTime-2.0 # seconds
            testColorChange = 8  # 8 is good for demonstration of same size and larger size adaptor  
        
        
        # change adaptor color with each time step to produce flicker
        self.adaptorColorChange = -self.gray # black
        if np.mod(self.timeCount, 2)== 0:
            self.adaptorColorChange=self.gray
                
        
        # if statements holding stimuli creation instructions
        if condition==0: # Crosses (Adapter size by divisible by 2 & 6)
            self.adaptorSize=42
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray  
            for i in np.arange(1,5): # for range 1-4 crosses
                self.centerPosition =[0,0]
                if i==1: # left
                   self.centerPosition = [self.i_x/2, self.i_y/4]
                if i==2: # right
                   self.centerPosition = [self.i_x/2, 3*self.i_y/4]
                if i==3: # top
                   self.centerPosition = [1*self.i_x/4, self.i_y/2]
                if i==4: # bottom
                   self.centerPosition = [3*self.i_x/4, self.i_y/2]   
                self.centerPosition[0] = np.round(self.centerPosition[0])
                self.centerPosition[1] = np.round(self.centerPosition[1])
                if time< self.testOnset: # draw adaptors   
                    # draw crosses
                    if i==3 or i==4:
                        # vertical
                        self.startInputImage[self.centerPosition[0]-self.adaptorSize/2-1:self.centerPosition[0]+self.adaptorSize/2:1, self.centerPosition[1]-self.adaptorSize/6-1:self.centerPosition[1]+self.adaptorSize/6:1]=self.gray+self.adaptorColorChange
                        # horizontal
                        self.startInputImage[self.centerPosition[0]-self.adaptorSize/6-1:self.centerPosition[0]+self.adaptorSize/6:1, self.centerPosition[1]-self.adaptorSize/2-1:self.centerPosition[1]+self.adaptorSize/2:1]=self.gray+self.adaptorColorChange
                        # make outline, by cutting middle
                        self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2 -1:1, self.centerPosition[1]-self.adaptorSize/6:self.centerPosition[1]+self.adaptorSize/6-1:1]=self.gray
                        self.startInputImage[self.centerPosition[0]-self.adaptorSize/6:self.centerPosition[0]+self.adaptorSize/6 -1:1, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2-1:1]=self.gray
                else: # draw test stimuli
                    testColor = self.gray+testColorChange
                    if i==1 or i==3:
                        testColor=self.gray-testColorChange
                    # vertical
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2-1:self.centerPosition[0]+self.adaptorSize/2:1, self.centerPosition[1]-self.adaptorSize/6-1:self.centerPosition[1]+self.adaptorSize/6:1]=testColor
                    # horizontal
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/6-1:self.centerPosition[0]+self.adaptorSize/6:1, self.centerPosition[1]-self.adaptorSize/2-1:self.centerPosition[1]+self.adaptorSize/2:1]=testColor   
            
        if condition==1: # Blob 
            self.adaptorSize=42
            self.centerPosition=[0,0]
            self.startInputImage = np.zeros((self.i_x, self.i_y))
            if time< self.testOnset:  # draw adaptors   
                # right (blurry square)
                self.centerPosition = [self.i_x/2, 3*self.i_y/4]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]= self.adaptorColorChange
                # blur entire image before putting other elements on
                temp = np.ones(10)/100;
                Blur = scipy.ndimage.convolve1d(self.startInputImage, temp)
                self.startInputImage = Blur + self.gray;  
                # left
                self.centerPosition = [self.i_x/2, self.i_y/4];
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
            else: # draw test stimuli
               self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray;
               self.centerPosition = [self.i_x/2, self.i_y/4];
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
               self.centerPosition = [self.i_x/2, 3*self.i_y/4];
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange    
            
        if condition==2: # SizeMatch 
            self.adaptorSize=42
            self.centerPosition=[0,0]
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors   
               # left
               self.adaptorSize = 54
               self.centerPosition = [round(2*self.i_x/3), self.i_y/4]
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
               # right
               self.adaptorSize = 30
               self.centerPosition = [round(2*self.i_x/3),3*self.i_y/4]
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
               # top
               self.adaptorSize = 42
               self.centerPosition = [self.i_x/4,2*self.i_y/4]
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
               self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray  
            else: # draw test stimuli
                # left
                self.centerPosition = [round(2*self.i_x/3), self.i_y/4]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                # right
                self.centerPosition = [round(2*self.i_x/3), 3*self.i_y/4]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                # top
                self.centerPosition = [self.i_x/4, self.i_y/2]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
            
        if condition==3: # Bipartite 
            self.adaptorSize=120
            self.centerPosition=[0,0]
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors   
                 # center vertical line
                 self.centerPosition = [self.i_x/2, self.i_y/2]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-1:self.centerPosition[1]+1]=self.gray+self.adaptorColorChange              
            else:
                 # draw test stimuli
                 # darker gray for background
                 self.startInputImage = (np.ones((self.i_x, self.i_y))*self.gray)-30 
                 # left side
                 self.centerPosition = [self.i_x/2, self.i_y/2]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]]=self.gray+testColorChange                  
                  # right side
                 self.centerPosition = [self.i_x/2, self.i_y/2]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]:self.centerPosition[1]+self.adaptorSize/2]=self.gray
        
        if condition==4: # Pyramids
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors
                for self.adaptorSize in np.arange(62,10,-16):
                     # right 
                     self.centerPosition = [self.i_x/2, 3*self.i_y/4]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                     # left
                     self.centerPosition = [self.i_x/2, self.i_y/4]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
            else: # draw test stimuli
                self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray
                # draw pyramids
                count=1
                for self.adaptorSize in np.arange(62,10,-16):
                    self.centerPosition = [self.i_x/2, self.i_y/4]
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange*count
                    self.centerPosition = [self.i_x/2, 3*self.i_y/4]
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange*count    
                    count=count+1
                count=count-1
                # draw comparison centers
                self.centerPosition = [self.i_x/4, self.i_y/4]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange*count
                self.centerPosition = [self.i_x/4, 3*self.i_y/4]
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange*count    
    
        if condition==5: # Annuli 
            self.centerPosition=[0, 0]
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors  (lines are thicker than for other demos) 
                 # right, top
                 self.centerPosition = [self.i_x/4, 3*self.i_y/4]
                 self.adaptorSize=42+2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # right, bottom
                 self.adaptorSize=18+2
                 self.centerPosition = [3*self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, top
                 self.adaptorSize=42+2
                 self.centerPosition = [self.i_x/4, self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, bottom
                 self.centerPosition = [3*self.i_x/4, self.i_y/4]
                 self.adaptorSize=18+2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray              
            else: # draw test stimuli                
                 # right, top
                 self.adaptorSize=42
                 self.centerPosition = [self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # right, bottom
                 self.adaptorSize=42
                 self.centerPosition = [3*self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, top
                 self.adaptorSize=42
                 self.centerPosition = [self.i_x/4, self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, bottom
                 self.centerPosition = [3*self.i_x/4, self.i_y/4]
                 self.adaptorSize=42
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange
                 self.adaptorSize=18;
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 
        if condition==6: # Incomplete 
            self.adaptorSize=84
            self.centerPosition=[0,0]
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors  (lines are thicker than for other demos) 
                 # right, top
                 self.centerPosition = [self.i_x/4, 3*self.i_y/4]
                 self.adaptorSize=42+2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # right, bottom
                 self.adaptorSize=18+2
                 self.centerPosition = [3*self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, top
                 self.adaptorSize=42+2
                 self.centerPosition = [self.i_x/4, self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, bottom
                 self.centerPosition = [3*self.i_x/4, self.i_y/4]
                 self.adaptorSize=18+2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                 self.adaptorSize= self.adaptorSize-2
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray              
            else: # draw test stimuli                
                 # right, top
                 self.adaptorSize=42
                 self.centerPosition = [self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # right, bottom
                 self.adaptorSize=42
                 self.centerPosition = [3*self.i_x/4, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, top
                 self.adaptorSize=42
                 self.centerPosition = [self.i_x/4, self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange
                 self.adaptorSize=18
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                 # left, bottom
                 self.centerPosition = [3*self.i_x/4, self.i_y/4]
                 self.adaptorSize=42
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray-testColorChange
                 self.adaptorSize=18;
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray     
           
        if condition==7: # Robinson-deSa 2013
            self.adaptorSize=42
            self.centerPosition=[0,0]
            self.startInputImage = np.zeros((self.i_x, self.i_y))
            if time< self.testOnset:  # draw adaptors   
                 self.startInputImage[:,:]= self.gray + self.adaptorColorChange
                 # right 
                 self.centerPosition = [self.i_x/2, 3*self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray
                 # left
                 self.centerPosition = [self.i_x/2, self.i_y/4]
                 self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray
            else: # draw test stimuli
                  # equal-sized test stimulus
                  self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray
                  self.centerPosition = [self.i_x/2, self.i_y/4];
                  self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+testColorChange
                  # small test stimulus
                  self.centerPosition = [self.i_x/2, 3*self.i_y/4];
                  self.startInputImage[self.centerPosition[0]-np.round(self.adaptorSize/4):self.centerPosition[0]+np.round(self.adaptorSize/4), self.centerPosition[1]-np.round(self.adaptorSize/4):self.centerPosition[1]+np.round(self.adaptorSize/4)]=self.gray+testColorChange    
            
        if condition==8: # Robinson-deSa2012-E1 (stimulus sizes multiply degrees by 10 for pixels)
            # (self.adaptorSize (100,40,20),self.testSize (20,100))
            self.adaptorSize=100
            self.testSize=100
            self.centerPosition=[0,0]
            self.startInputImage = np.zeros((self.i_x, self.i_y))
            if time< self.testOnset:  # draw adaptors   
                self.startInputImage[:,:]= self.gray
                 # centered 
                self.centerPosition = [self.i_x/2, self.i_y/2];
                self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]= self.gray + self.adaptorColorChange
            else: # draw test stimulus
                 self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray
                 # left
                 self.centerPosition = [self.i_x/2, self.i_y/2];
                 self.startInputImage[self.centerPosition[0]-self.testSize/2:self.centerPosition[0]+self.testSize/2, self.centerPosition[1]-self.testSize/2:self.centerPosition[1]+self.testSize/2]=self.gray+testColorChange;
    
        if condition==9: # Robinson-deSa2012-E2 (stimulus sizes multiply degrees by 10 for pixels)
            # (self.adaptorSize, middleAdaptorSize, self.innerAdaptorSize,self.testSize)
             self.adaptorSize=140
             self.middleAdaptorSize=100
             self.innerAdaptorSize=0
             self.testSize=100
             self.centerPosition=[0,0]
             self.startInputImage = np.zeros((self.i_x, self.i_y))
             if time< self.testOnset:  # draw adaptors   
                  self.startInputImage[:,:]= self.gray
                  # centered 
                  self.centerPosition = [self.i_x/2, self.i_y/2]
                  # outer edge
                  # upper left
                  self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0], self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]]= self.gray + self.adaptorColorChange
                  # bottom left
                  self.startInputImage[self.centerPosition[0]:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]]= self.gray - self.adaptorColorChange
                  # upper right
                  self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0], self.centerPosition[1]:self.centerPosition[1]+self.adaptorSize/2]= self.gray - self.adaptorColorChange
                  # bottom right
                  self.startInputImage[self.centerPosition[0]:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]:self.centerPosition[1]+self.adaptorSize/2]= self.gray + self.adaptorColorChange
                  # middle edge
                  # upper left
                  self.startInputImage[self.centerPosition[0]-self.self.middleAdaptorSize/2:self.centerPosition[0], self.centerPosition[1]-self.self.middleAdaptorSize/2:self.centerPosition[1]]= self.gray - self.adaptorColorChange
                  # bottom left
                  self.startInputImage[self.centerPosition[0]:self.centerPosition[0]+self.self.middleAdaptorSize/2, self.centerPosition[1]-self.self.middleAdaptorSize/2:self.centerPosition[1]]= self.gray + self.adaptorColorChange
                  # upper right
                  self.startInputImage[self.centerPosition[0]-self.self.middleAdaptorSize/2:self.centerPosition[0], self.centerPosition[1]:self.centerPosition[1]+self.self.middleAdaptorSize/2]= self.gray + self.adaptorColorChange
                  # bottom right
                  self.startInputImage[self.centerPosition[0]:self.centerPosition[0]+self.self.middleAdaptorSize/2, self.centerPosition[1]:self.centerPosition[1]+self.self.middleAdaptorSize/2]= self.gray - self.adaptorColorChange
                  # gray interior
                  if self.innerAdaptorSize>0:
                      self.startInputImage[self.centerPosition[0]-self.innerAdaptorSize/2:self.centerPosition[0]+self.innerAdaptorSize/2, self.centerPosition[1]-self.innerAdaptorSize/2:self.centerPosition[1]+self.innerAdaptorSize/2]= self.gray
             else: # draw test stimulus
                   self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray
                   # left
                   self.centerPosition = [self.i_x/2, self.i_y/2]
                   self.startInputImage[self.centerPosition[0]-self.testSize/2:self.centerPosition[0]+self.testSize/2, self.centerPosition[1]-self.testSize/2:self.centerPosition[1]+self.testSize/2]=self.gray+testColorChange 
                
        if condition==10: # Prediction 
            self.testSize=50
            self.centerPosition=[0,0]
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            if time< self.testOnset:  # draw adaptors 
                # illusory contour on left
                for self.adaptorSize in np.arange(32,5,-8):
                    # topright 
                    self.centerPosition = [3*self.i_x/8, self.i_y/2-self.testSize/2 - self.testSize];
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                    # leftright
                    self.centerPosition = [3*self.i_x/8, self.i_y/2+self.testSize/2- self.testSize];
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
        
                    # bottomright 
                    self.centerPosition = [3*self.i_x/8+self.testSize, self.i_y/2-self.testSize/2- self.testSize]
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                    # bottomleft
                    self.centerPosition = [3*self.i_x/8+self.testSize, self.i_y/2+self.testSize/2- self.testSize]
                    self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                    self.startinputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray  
                # middle gray  
                tempinputImage = np.ones((self.i_x, self.i_y))*self.gray
                # copy and paste from temp to startinput
                self.centerPosition = [self.i_x/2, self.i_y/2- self.testSize]
                self.startInputImage[self.centerPosition[0]-self.testSize/2+1:self.centerPosition[0]+self.testSize/2 -1, self.centerPosition[1]-self.testSize/2+1:self.centerPosition[1]+self.testSize/2-1]=tempinputImage[self.centerPosition[0]-self.testSize/2+1:self.centerPosition[0]+self.testSize/2 -1, self.centerPosition[1]-self.testSize/2+1:self.centerPosition[1]+self.testSize/2-1]                   
        
                # drawn contour on right
                for self.adaptorSize in np.arange(32,5,-8):
                    # topright 
                     self.centerPosition = [3*self.i_x/8, self.i_y/2-self.testSize/2 + self.testSize]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                     # leftright
                     self.centerPosition = [3*self.i_x/8, self.i_y/2+self.testSize/2+ self.testSize]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
         
                     # bottomright 
                     self.centerPosition = [3*self.i_x/8+self.testSize, self.i_y/2-self.testSize/2+ self.testSize]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                     # bottomleft
                     self.centerPosition = [3*self.i_x/8+self.testSize, self.i_y/2+self.testSize/2+ self.testSize]
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2:self.centerPosition[0]+self.adaptorSize/2, self.centerPosition[1]-self.adaptorSize/2:self.centerPosition[1]+self.adaptorSize/2]=self.gray+self.adaptorColorChange
                     self.startInputImage[self.centerPosition[0]-self.adaptorSize/2+1:self.centerPosition[0]+self.adaptorSize/2 -1, self.centerPosition[1]-self.adaptorSize/2+1:self.centerPosition[1]+self.adaptorSize/2-1]=self.gray
                     
                # draw line 
                self.centerPosition = [self.i_x/2, self.i_y/2 + self.testSize]
                self.startInputImage[self.centerPosition[0]-self.testSize/2+1:self.centerPosition[0]+self.testSize/2 , self.centerPosition[1]-self.testSize/2+1:self.centerPosition[1]+self.testSize/2]=self.gray+self.adaptorColorChange
                self.startInputImage[self.centerPosition[0]-self.testSize/2+2:self.centerPosition[0]+self.testSize/2 -1 , self.centerPosition[1]-self.testSize/2+2:self.centerPosition[1]+self.testSize/2-1]=self.gray
               
            else: # draw test stimuli
                self.startInputImage =  np.ones((self.i_x, self.i_y))*self.gray
                #left gray
                self.centerPosition = [self.i_x/2, self.i_y/2 - self.testSize]
                self.startInputImage[self.centerPosition[0]-self.testSize/2+1:self.centerPosition[0]+self.testSize/2 -1, self.centerPosition[1]-self.testSize/2+1:self.centerPosition[1]+self.testSize/2-1]=self.gray+testColorChange
                #right gray
                self.centerPosition = [self.i_x/2, self.i_y/2 + self.testSize]
                self.startInputImage[self.centerPosition[0]-self.testSize/2+1:self.centerPosition[0]+self.testSize/2 -1, self.centerPosition[1]-self.testSize/2+1:self.centerPosition[1]+self.testSize/2-1]=self.gray+testColorChange 
        
        if condition == 11: #Whites illusion
            self.startInputImage = np.ones((self.i_x, self.i_y))*self.gray
            stim, mask_dark, mask_bright = wi.evaluate(self.patch_h,self.direction)
            if time< self.testOnset: # Show adaptors (mask)
                if self.adaptorColorChange == self.gray:
                    self.startInputImage= mask_bright
                else:
                    self.startInputImage= mask_dark
            else: # Show test stimuli
                self.startInputImage = stim
        
        
        # fixation markers
        self.startInputImage[self.i_x/2-2,self.i_y/2-2]=255
        self.startInputImage[self.i_x/2  ,self.i_y/2  ]=255
        self.startInputImage[self.i_x/2-2,self.i_y/2  ]=0
        self.startInputImage[self.i_x/2  ,self.i_y/2-2]=0
                  
        # Convert RGB input image to red-green, blue-yellow, white-black coordinates
        self.inputImage = np.zeros((200,200,3))
        self.inputImage[:,:,0] = self.startInputImage
        self.inputImage[:,:,1] = self.startInputImage
        self.inputImage[:,:,2] = self.startInputImage
        out = ConvertRGBtoOpponentColor(self.inputImage, self.gray)
        self.rg=out[0]
        self.by=out[1]
        self.wb=out[2]
        
            
    def LGNcells(self):
        """
        Usage:
        >>> LGNCells(self)
        
        "white-black center-surround processing (but with current parameters it actually does very little)"
        Appears to normalize the image, often a step introduced to make edges more easily extractable.
        
        
        Parameters
        ----------
        wb : array_like
                white-black color channel of input image [400x400 array]
        C/E : array_like
                Gaussian pulses [typ. 23x23,7x7]       
        
        
        Returns
        ----------
        LGNwb : array_like
                LGN cell processed output for white-black color channel of input image [200x200 array]
        wb,rg,by : array_like
                Padded color channels of input image [400x400 arrays]
        PaddingSize : int
                Size of padding required for this image. [typ. 100]
        
        """
        # padding
        self.PaddingSize = math.floor(np.max(self.wb.shape)/2) 
        PaddingColor = self.wb[0,0]
        self.wb2 = im_padding(self.wb, self.PaddingSize, PaddingColor)
        
        # convolution - reflection of each other
        OnOff_Excite =  conv2(self.wb2, self.C, mode='same') 
        OnOff_Inhibit = conv2(self.wb2, self.E, mode='same')
        
        # shunting
        paramA = 50 # 1 
        paramB = 90 # 90
        paramD = 60 # 60
        
        # OnOff Excitatory/Inhibitory processing calculation
        x_OnOff = (paramB*OnOff_Excite-paramD*OnOff_Inhibit)/(paramA+(OnOff_Excite+OnOff_Inhibit))
        x_OffOn = (paramB*OnOff_Inhibit-paramD*OnOff_Excite)/(paramA+(OnOff_Inhibit+OnOff_Excite))
        x_pos = x_OnOff - x_OffOn
        x_neg = x_OffOn - x_OnOff
        
        # half-wave rectify
        x_pos[x_pos<0] = 0
        x_neg[x_neg<0] = 0
        
        # LGN cell out
        self.LGNwb = x_pos - x_neg
        
        # pad planes for all color channels for later use
        PaddingColor = self.wb[0,0]
        self.wb = im_padding(self.wb, self.PaddingSize, PaddingColor)
        PaddingColor = self.rg[0,0]
        self.rg = im_padding(self.rg, self.PaddingSize, PaddingColor)
        PaddingColor = self.by[0,0]
        self.by = im_padding(self.by, self.PaddingSize, PaddingColor)
        
        
    
    def simpleCell(self) :
        """
        Usage:
        >>> simpleCell(self)
        
        Theoretical replication of simple cell processing of LGN cell processed image
        "Orientations are only based on inputs from the white-black color channel"
        Involves convolution, padding and half wave rectification for each polarity,
        this is essentially the edge detection mechanism of the model.
        
        Parameters
        -----------
        LGNwb : array_like
                LGN cell white-black color channel processed image [200x200 array] 
        F : array_like
                Orientation filters ("ODOG") [200x200 array] 
        PaddingSize : int
                Size of padding [typ. 100] 
        K : int
                Number of polarities [4 horizonal, 4 vertical]
        
        Returns
        -----------
        y : array_like
                simple cell processed output image [200x200x4] or boundaries of input
        
        """
        y = np.zeros((self.i_x,self.i_y,self.K))

        for i in range(self.K):
            Ini = np.abs(conv2(self.LGNwb, self.F[:,:,i]))   # convolve
            Ini = im_cropping(Ini, self.PaddingSize)         # padding
            Ini[Ini<0] = 0                                   # half wave rectify
            y[:,:,i]=Ini
            
        self.y = y
        
        
    def complexCell(self, boundaryUpperLimit):
        """
        Usage: 
        >>> complexCell(self,25)
        
        Theoretical replication of complex cell processing of simple cell processed image
        Polarities are summed into two directions (up/down and left/right), boundary limit set,
        and the tonic input is added to each orientation.
        
        Parameters
        -----------
        y :
                simple cell processed output image
        K :
                number of polarities
        boundaryUpperLimit : 
                upper limit for boundary activity 
        inI : 
                tonic input
        nOrient :
                number of orientations        
        
        Returns
        -----------
        w1 : 
                complex cell processed output image
        """
        
        # pool across contrast polarity
        z1= np.zeros((self.y.shape[0], self.y.shape[1], self.nOrient))
        for k in np.arange(0,self.K/2):
            z1[:,:,k] = self.y[:,:,k] + self.y[:,:,k+self.K/2]
        
        # limit with boundary upper limit
        z1[z1>boundaryUpperLimit] = boundaryUpperLimit
        
        # Add tonic input, inI, to boundaries
        self.w1= np.zeros((self.y.shape[0], self.y.shape[1], self.nOrient))
        for k in np.arange(0,self.nOrient):
            self.w1[:,:,k] = self.inI  + z1[:,:,k]
        
    
    def gateComp(self,time, Agate,Bgate,Cgate,Rhogate):
        """
        Usage: 
        >>> gateComp(self, time, 20.0, 1.0, 1.0, 0.007)        
        
        Orientation specific habituating transmitter gate. Initialised on first time step
        and then gradually depletes over time.
        
        Parameters
        -----------
        time : float
                current time in the simulation        
        'A/B/C/Rho'gate :
                Gating parameters
        w1 :
                Output of complex cell
        
        Returns
        -----------
        gate :
                image of the gating structure [200x200x2]
        
        
        """   
        
        # initialize gate on first time step
        if time==self.startTime :
            self.gate = Agate/(Bgate + Cgate*self.inI) * np.ones(self.w1.shape)
                
        # identifsy equilibrium solution to gate
        gate_equil = Agate/(Bgate + Cgate* self.w1)
        
        # habituating gate solution
        self.gate = gate_equil + (self.gate - gate_equil)* np.exp(-Rhogate*(Bgate+Cgate*self.w1)*self.timeStep)
            
    def dipoleComp(self):
        """
        Usage:
        >>> dipleComp(self)
        
        Cross orientation dipole competition
               
        
        Parameters
        ----------
        gate :
                gate 
        w1 :
                complex cell output
        gdAcrossWeight :
                strength of inhibition between orthogonal orientations
        bThresh :
                boundary threshold
        
        Returns
        ----------
        O2 :
                processed output prior to filling in stage
        
        """
        gdAcrossWeight = 0.5 # cross orientation weighting
                
        # gating applied to boundary signal and orthogonal orientation inhibition applied
        v_1 = self.gate[:,:,0]*self.w1[:,:,0] - gdAcrossWeight*self.gate[:,:,1]*self.w1[:,:,1] 
        v_2 = self.gate[:,:,1]*self.w1[:,:,1] - gdAcrossWeight*self.gate[:,:,0]*self.w1[:,:,0] 
        
        # half-wave rectify
        v_1[v_1<0] = 0  
        v_2[v_2<0] = 0
        
        O1=np.zeros((v_1.shape[1],v_1.shape[0],self.nOrient))
        O1[:,:,0] = v_1
        O1[:,:,1] = v_2
        
        # soft threshold for boundaries
        bThresh=9.5
        self.O2=O1-bThresh
        self.O2[self.O2<0] = 0   
        
    
    def fillingin(self):
        """
        Usage: 
        >>> fillingin(self)
        
        FIDO - regions of connected boundary points    
        "Most of this code is an algorithmic way of identifying distinct Filling-In DOmains (FIDOs)"
        
        Parameters
        -----------
        O2 :
                
        BndThr :
                
        nOrient :
                
        Bshift1/2 :
                Boundaries that block flow for the corresponding shift
        shiftOrientation :
                orientation of flow 
        shift :
                up, down, left, right
        
        Returns
        -----------
        P :
                boundary structure
        FIDO :
                fido
        S_rb/by/wb :
                output signals
        
        """
        shiftOrientation = [np.pi/2, np.pi/2, 0, 0]
        # Boundaries that block flow for the corresponding shift
        Bshift1 = np.array([[-1, -1],[ 0, -1],[ -1, -1],[ -1, 0]])
        Bshift2 = np.array([[-1,  0],[ 0,  0],[  0, -1],[  0, 0]])
         
        # boundary signals
        # Most of this code is an algorithmic way of identifying distinct Filling-In DOmains (FIDOs)
        BndSig = np.sum(self.O2[:,:,:],2)
        thint = self.O2.shape
        BndSig = 100*BndSig
        BndSig[BndSig < 0] = 0
        
        sX = np.size(BndSig, 0)
        sY = np.size(BndSig, 1)
        
        stimarea_x = np.arange(1,np.size(BndSig, 0)-1) 
        stimarea_y = np.arange(1,np.size(BndSig, 1)-1)
        
        # Setting up boundary structures
        P=np.zeros((sX,sY,4))
        for i in np.arange(0,4):
            dummy = np.ones((sX, sY))
            p1 = stimarea_x + Bshift1[i,0]
            q1 = stimarea_y + Bshift1[i,1]
            p2 = stimarea_x + Bshift2[i,0]
            q2 = stimarea_y + Bshift2[i,1]
            
            currentBoundary = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            currentBoundary1 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            currentBoundary2 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
            
            # for both orientations at each polarity
            a1=np.abs(np.sin(shiftOrientation[i] - 0))
            currentBoundary1[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1] = a1*(self.O2[p1[0]:p1[-1]+1:1,q1[0]:q1[-1]+1:1,0] + self.O2[p2[0]:p2[-1]+1:1,q2[0]:q2[-1]+1:1,0] )
            
            a2=np.abs(np.sin(shiftOrientation[i] - np.pi/2))
            currentBoundary2[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1] = a2*(self.O2[p1[0]:p1[-1]+1:1,q1[0]:q1[-1]+1:1,1] + self.O2[p2[0]:p2[-1]+1:1,q2[0]:q2[-1]+1:1,1] )
            
            currentBoundary=currentBoundary1+currentBoundary2
            a = currentBoundary
            a[a>0] = 1
            a1=dummy[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1]
            a2=    a[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1]
            
            P[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1,i] =   a1- a2
                
        # find FIDOs and average within them
        FIDO_ini = np.zeros((sX, sY))
                
        # unique number for each cell in the FIDO
        for i in np.arange(0,sX):
            for j in np.arange(0,sY):
                FIDO_ini[i,j] = (i+1)+ (j+1)*thint[0]  
        
        # Grow each FIDO so end up with distinct domains with a common assigned number
        FIDO_edit=FIDO_ini
        n = 500 # optimization parameter, number of growth steps (Previous while loop with FIDO_edit==FIDO_ini)
        for n in np.arange(1,500):
            FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1-1:199-1,1:199]*P[1:199, 1:199,0] ) 
            FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1+1:199+1,1:199]*P[1:199, 1:199,1] ) 
            FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1:199,1-1:199-1]*P[1:199, 1:199,2] ) 
            FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1:199,1+1:199+1]*P[1:199, 1:199,3] ) 
        
        FIDO=FIDO_edit    
        
        # input is color signals
        wbColor = im_cropping(self.wb, self.PaddingSize)
        rgColor = im_cropping(self.rg, self.PaddingSize)
        byColor = im_cropping(self.by, self.PaddingSize)
        
        # Filling-in values for white-black, red-green, and blue-self.yellow
        self.S_wb = np.zeros((sX, sY))
        self.S_rg = np.zeros((sX, sY))
        self.S_by = np.zeros((sX, sY)) 
        
        
        # Compute average color for unique FIDOs
        FIDO = FIDO.astype(int)
        labels = np.arange(FIDO.max()+1, dtype=int)
        self.S_wb = labeled_mean(wbColor, FIDO, labels)[FIDO]
        self.S_rg = labeled_mean(rgColor, FIDO, labels)[FIDO]
        self.S_by = labeled_mean(byColor, FIDO, labels)[FIDO]
        

    def saveImages(self,condition):
        """
        Usage:
        >>> saveImages(self,condition)    
        
        Save image files of network behaviour
        
        """
        
        # Create directorself.y for results
        self.resultsDirectory = os.path.dirname("{0}/{1}".format("Condition",
                                           self.Conditions[condition]))
        if os.path.exists(self.resultsDirectory)==0:
            os.mkdir(self.resultsDirectory)        
        
        
        # Make boundarself.y animated gif
        self.orientedImage = np.ones((self.i_x, self.i_y,3))  
        step = 1   # can sub-sample image if bigger than 1 
                    
        # transform boundarself.y values into intensitself.y signals for viewing image 
        for i in np.arange(0,self.i_x,step): # -step SPLINT
            for j in np.arange(0,self.i_y,step): # -step SPLINT
                # if vertical edge at this pixel, color it green
                if self.O2[i,j,1] >0:
                    ratio = self.O2[i,j,1]/80
                    if ratio<0.2:
                        ratio = 0.2
                    self.orientedImage[i, j,2] = 1-ratio # reduce blue -k SPLINT
                    self.orientedImage[i, j,0] = 1-ratio # reduce red  -k SPLINT
        
                # if horizontal edge at this pixel, color it blue
                if self.O2[i,j,0] >0:
                    ratio = self.O2[i,j,0]/80
                    if ratio<0.2:
                        ratio = 0.2
                    self.orientedImage[i, j, 1] = 1-ratio # reduce green
                    self.orientedImage[i, j, 0] = 1-ratio # reduce red
          
        
        # Convert values in the color FIDOs to something that can be presented in an image
        S_rgmax = np.max(np.max(np.abs(self.S_rg[:,:])))
        S_bymax = np.max(np.max(np.abs(self.S_by[:,:])))
        S_wbmax = np.max(np.max(np.abs(self.S_wb[:,:])))
        S_max1  = np.maximum(S_rgmax, S_bymax)
        S_max   = np.maximum(S_max1 , S_wbmax)
        
        # Convert FIDO values to self.self.rgB values (relative to self.graself.y and maximum FIDO value)
        S_rgb = ConvertOpponentColortoRGB(self.S_rg[:,:], self.S_by[:,:], self.S_wb[:,:], self.gray, S_max)
        # scale to 0 to 255 self.self.rgB values
        temp = 255.0* (S_rgb[:,:,0]/np.max(np.max(np.max(S_rgb))))
        
        # Make image of input, boundaries, and filled-in values to save as a png file
        self.thing = np.ones((self.i_x, 3*self.i_y, 3))
        
        # Input image on left (Participant Image)
        self.thing[:,0:self.i_y,:]=self.inputImage/255
        
        # Filled-in values on right (Computer Image)
        self.thing[:,2*self.i_y:3*self.i_y,0]=temp/255 
        self.thing[:,2*self.i_y:3*self.i_y,1]=temp/255  
        self.thing[:,2*self.i_y:3*self.i_y,2]=temp/255
        
        # Boundaries in center (Boundarself.y Image)
        self.thing[:,self.i_y:2*self.i_y,:]=self.orientedImage # +1 removed from self.y start SPLINT
        
        # Write individual frame files (with leading zero if less than 10)
        if self.timeCount>=10:
            filename = "{0}/{1}{2}{3}".format(self.resultsDirectory, 'All',self.timeCount,'.png')
        else:
            filename = "{0}/{1}{2}{3}".format(self.resultsDirectory,'All0',self.timeCount,'.png')
        
        #Same image to file
        scipy.misc.imsave(filename, self.thing)


def ConvertOpponentColortoRGB(rg, by, wb, gray, maxi):
    """ 
    This function converts an array of color opponent values into RGB values
    
    Parameters
    -----------
    rg/by/wb : array_like
            Seperate RGB color channels of input image
    gray : int
            integer value of gray 
    max : int 
            maximum color value
            
    Returns
    -----------
    rgb : array_like
            Combined color channels into RGB
    """
    arraySize = rg.shape   
    rgb = np.zeros((arraySize[0], arraySize[1], 3))
    rgb[:,:,1] = gray*(3*wb - 3*rg - 2*by)/(3*maxi) # green
    rgb[:,:,0] = gray*2*rg/maxi + rgb[:,:,1] # red
    rgb[:,:,2] = gray*2*by/maxi + (rgb[:,:,0] + rgb[:,:,1] )/2 # blue
    rgb = rgb + gray
    rgb[rgb<0] = 0
    rgb[rgb>255] = 255
    return rgb
    
def  ConvertRGBtoOpponentColor(rgb, gray):
    """
    This function converts an array of RGB values into color opponent values
    
    Parameters
    -----------
    rgb : array_like
            Combined color channels into RGB
    gray : int
            integer value of gray 
            
    Returns
    -----------
    rg/by/wb : array_like
            Seperate RGB color channels of input image
   
    """
    rgb = rgb - gray
    rgb[rgb>127] = 127
    wb = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/ (3*gray) 
    rg = (rgb[:,:,0] - rgb[:,:,1])/(2*gray) 
    by = (rgb[:,:,2] - (rgb[:,:,0] + rgb[:,:,1])/2)/(2*gray) 
    return [rg, by,wb]
    
def im_cropping (*args):
    """ 
    Image cropping
    
    """
    InputMatrix = args[0]
    nPixCut = args[1]
    [x, y] = InputMatrix.shape
    StimX = x - 2 * nPixCut
    StimY = y - 2 * nPixCut
    # crop the actual stimulus area 
    output_matrix = InputMatrix[(nPixCut):(nPixCut+StimX), (nPixCut):(nPixCut+StimY)] 
    return output_matrix

def im_padding (*args):
    """
    Image padding
    
    """
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
    """
    Produces a 2D Gaussian pulse    
    
    Parameters
    ----------
    GCenter : int
            Centre point of Gaussian pulse
    Gamp : int
            Amplitude of Gaussian pulse
    Ggamma : int
            FWHM of Gaussian pulse
    Gconst : float
            Unkown parameter of density function
    
    Returns
    ----------
    GKernel : array_like
            Gaussian kernel
    """
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
    Emulate the Matlab function conv2 from Mathworks.
    
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
