# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:26:40 2015
Translated Python Version of Gregory Francis' contour adaptation
"""

# From the commnand line, go to the directory with the images and type
# convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif

import numpy as np
import math
import os
from scipy.ndimage.measurements import mean as labeled_mean
import scipy
import matplotlib.pyplot as plt

# Side functions
import image_edit

# Code to make GIFs or make PNGS in file
makeAnimatedGifs=0 # 0- make PNGs, 1- make GIFs

# Parameters
gray = 127

# Filling-in stage paramters  
# Boundaries are shifted in a grid plane relative to color/brightness signals
shift = np.array([[-1,0],[1,0],[0,-1],[0, 1]]) # up, down, left, right
shiftOrientation = [np.pi/2, np.pi/2, 0, 0] # orientation of flow
# Boundaries that block flow for the corresponding shift
Bshift1 = np.array([[-1, -1],[ 0, -1],[ -1, -1],[ -1, 0]])
Bshift2 = np.array([[-1,  0],[ 0,  0],[  0, -1],[  0, 0]])

# Image size
i_x=200
i_y=200

################  Define LGN kernels and other paramters  #####################

# excitatory condition
C_ = 10 # coefficient (C>E) 18
alpha = .5 # radius of spreads (a<b) .5
Gconst = image_edit.floatingpointtointeger(4, 2*np.log(2)) # avoid floating point errors
C = image_edit.Gaussian2D([0,0], C_, alpha, Gconst)

# inhibitory condition
E_ = .5  # coefficient
beta = 2 # radius of spreads
E = image_edit.Gaussian2D([0,0], E_, beta, Gconst)

# Parameters for Orientation Gated Dipoles
Agate = 20.0  
Bgate = 1.0
Cgate = 1.0
Rhogate = 0.007  
inI = 5 

# number of polarities (4 means horizontal and vertical orientations)
K = 4  
# number of orientations for simple cells
nOrient = K/2  
# boundaries are shifted relative to image plane(boundaries are between pixels)
orientationShift = 0.5  

gamma = 1.75
G = image_edit.Gaussian2D([0+orientationShift,0+orientationShift], 1, gamma, 2)
F = np.zeros((G.shape[0],G.shape[1],4))

# Orientation filters (Difference of Offset Gaussians)
for k in np.arange(0,K):
    m = np.sin((2*np.pi*(k+1))/K)
    n = np.cos((2*np.pi*(k+1))/K)
    H =image_edit.Gaussian2D([m+orientationShift,n+orientationShift],1,gamma,2)
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
    
    F[:,:,k] = posF + negF
    
    # normalize full kernel
    normalizer = np.sum(np.sum( F[:,:,k]*F[:,:,k] ) )
    F[:,:,k] = F[:,:,k]/np.sqrt(normalizer)




################################## CREATE INPUT STIMULUS ####################################### 

# Names of various stimulus conditions
Conditions = ['Crosses', 'Blob', 'SizeMatch', 'Bipartite', 'Pyramids',
              'Annuli', 'Incomplete', 'Robinson-deSa2013', 
              'Robinson-deSa2012-E1', 'Robinson-deSa2012-E2', 'Prediction']

#for condition in np.arange(1,12):
# Only consider single condition for speed
condition = 0

print "Simulation condition : ", Conditions[condition]

# Create directory for results
resultsDirectory = os.path.dirname("{0}/{1}".format("Condition",
                                   Conditions[condition]))
if os.path.exists(resultsDirectory)==0:
    os.mkdir(resultsDirectory)

# Set timing Parameters (seconds)
timeStep = 0.1 # Originally 0.1 seconds (Changed for speed)
stopTime = 6 # seconds 6 for A&G conditions - makes for 4 second adaptation
startTime = timeStep
testOnset = stopTime-2.0 
# 5 is typical for A&G conditions (this is the change, increase or decrease,
# from the gray background)
testColorChange=5 

if condition==7:  # special settings for R&dS 2013 condition
    timeStep = 0.153 # seconds
    stopTime = 8.12 # seconds -- 6.12 seconds of adaptation
    #stopTime = 2+timeStep # seconds - no adapt condition (actually one presentation) -- 0.153 seconds of adaptation
    startTime = timeStep
    testOnset = stopTime-2.0 # seconds
    testColorChange = 8  # 8 is good for demonstration

if condition==8 or condition==9: # special settings for R&dS 2012 conditions
    timeStep = 0.16 # seconds
    stopTime = 7.0 # seconds -- 5 seconds of adaptation (R&dS 2012 had long initial period and then short re-adapt)
    #stopTime = 2+3*timeStep # seconds -no adapt condition (actually one flicker cycle of adaptation)
    startTime = timeStep
    testOnset = stopTime-2.0 # seconds
    testColorChange = 8  # 8 is good for demonstration of same size and larger size adaptor  

# Initiate time sequence
timeCount=0
for t in np.arange(startTime,stopTime+timeStep,timeStep):
    #t = testOnset
    
    timeCount= timeCount+1
    
    # change adaptor color with each time step to produce flicker
    adaptorColorChange = -gray # black
    if np.mod(timeCount, 2)== 0:
        adaptorColorChange=gray
       
    time=t
       
    if condition==0: # Crosses (Adapter size by divisible by 2 & 6)
        adaptorSize=42
        startinputImage = np.ones((i_x, i_y))*gray  
    for i in np.arange(1,5): # for range 1-4 crosses
        centerPosition =[0,0]
        if i==1: # left
           centerPosition = [i_x/2, i_y/4]
        if i==2: # right
           centerPosition = [i_x/2, 3*i_y/4]
        if i==3: # top
           centerPosition = [1*i_x/4, i_y/2]
        if i==4: # bottom
           centerPosition = [3*i_x/4, i_y/2]   
        centerPosition[0] = np.round(centerPosition[0])
        centerPosition[1] = np.round(centerPosition[1])
        if time< testOnset: # draw adaptors   
            # draw crosses
            if i==3 or i==4:
                # vertical
                startinputImage[centerPosition[0]-adaptorSize/2-1:centerPosition[0]+adaptorSize/2:1, centerPosition[1]-adaptorSize/6-1:centerPosition[1]+adaptorSize/6:1]=gray+adaptorColorChange
                # horizontal
                startinputImage[centerPosition[0]-adaptorSize/6-1:centerPosition[0]+adaptorSize/6:1, centerPosition[1]-adaptorSize/2-1:centerPosition[1]+adaptorSize/2:1]=gray+adaptorColorChange
                # make outline, by cutting middle
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2 -1:1, centerPosition[1]-adaptorSize/6:centerPosition[1]+adaptorSize/6-1:1]=gray
                startinputImage[centerPosition[0]-adaptorSize/6:centerPosition[0]+adaptorSize/6 -1:1, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2-1:1]=gray
        else: # draw test stimuli
            testColor = gray+testColorChange
            if i==1 or i==3:
                testColor=gray-testColorChange
            # vertical
            startinputImage[centerPosition[0]-adaptorSize/2-1:centerPosition[0]+adaptorSize/2:1, centerPosition[1]-adaptorSize/6-1:centerPosition[1]+adaptorSize/6:1]=testColor
            # horizontal
            startinputImage[centerPosition[0]-adaptorSize/6-1:centerPosition[0]+adaptorSize/6:1, centerPosition[1]-adaptorSize/2-1:centerPosition[1]+adaptorSize/2:1]=testColor   
    
    if condition==1: # Blob 
        adaptorSize=42
        centerPosition=[0,0]
        startinputImage = np.zeros((i_x, i_y))
        if time< testOnset:  # draw adaptors   
            # right (blurry square)
            centerPosition = [i_x/2, 3*i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]= adaptorColorChange
            # blur entire image before putting other elements on
            temp = np.ones(10)/100;
            Blur = scipy.ndimage.convolve1d(startinputImage, temp)
            startinputImage = Blur + gray;  
            # left
            centerPosition = [i_x/2, i_y/4];
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
            startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
        else: # draw test stimuli
           startinputImage =  np.ones((i_x, i_y))*gray;
           centerPosition = [i_x/2, i_y/4];
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
           centerPosition = [i_x/2, 3*i_y/4];
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange    
        
    if condition==2: # SizeMatch 
        adaptorSize=24
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors   
           # left
           adaptorSize = 54
           centerPosition = [round(2*i_x/3), i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
           # right
           adaptorSize = 30
           centerPosition = [round(2*i_x/3),3*i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
           # top
           adaptorSize = 42
           centerPosition = [i_x/4,2*i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray  
        else: # draw test stimuli
            # left
            centerPosition = [round(2*i_x/3), i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
            # right
            centerPosition = [round(2*i_x/3), 3*i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
            # top
            centerPosition = [i_x/4, i_y/2]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
        
    if condition==3: # Bipartite 
        adaptorSize=120
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray;
        if time< testOnset:  # draw adaptors   
           # left
           adaptorSize = 54
           centerPosition = [round(2*i_x/3), i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
           # right
           adaptorSize = 30
           centerPosition = [round(2*i_x/3),3*i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
           # top
           adaptorSize = 42
           centerPosition = [i_x/4,2*i_y/4]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
           startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray  
        else: # draw test stimuli
            # left
            centerPosition = [round(2*i_x/3), i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
            # right
            centerPosition = [round(2*i_x/3), 3*i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
            # top
            centerPosition = [i_x/4, i_y/2]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
    
    if condition==4: # Pyramids
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors   
             # center vertical line
             centerPosition = [i_x/2, i_y/2]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]:centerPosition[1]]=gray+adaptorColorChange;              
        else:
             # draw test stimuli
             # darker gray for background
             startinputImage = (np.ones((i_x, i_y))*gray)-30 
             # left side
             centerPosition = [i_x/2, i_y/2]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]]=gray+testColorChange                  
              # right side
             centerPosition = [i_x/2, i_y/2]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]:centerPosition[1]+adaptorSize/2]=gray
        
    if condition==5: # Annuli 
        startinputImage = create_stimulus.condition6([0, 0],i_x,i_y,t,testOnset,gray,testColorChange,adaptorColorChange)
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors
            for adaptorSize in np.arange(62,10,-16):
                 # right 
                 centerPosition = [i_x/2, 3*i_y/4]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                 # left
                 centerPosition = [i_x/2, i_y/4]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
        else: # draw test stimuli
            startinputImage =  np.ones((i_x, i_y))*gray
            # draw pyramids
            count=1
            for adaptorSize in np.arange(62,10,-16):
                centerPosition = [i_x/2, i_y/4]
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray-testColorChange*count
                centerPosition = [i_x/2, 3*i_y/4]
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange*count    
                count=count+1
            count=count-1
            # draw comparison centers
            centerPosition = [i_x/4, i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray-testColorChange*count
            centerPosition = [i_x/4, 3*i_y/4]
            startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange*count    
       
    if condition==6: # Incomplete 
        adaptorSize=84
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors  (lines are thicker than for other demos) 
             # right, top
             centerPosition = [i_x/4, 3*i_y/4]
             adaptorSize=42+2
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
             adaptorSize= adaptorSize-2
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # right, bottom
             adaptorSize=18+2
             centerPosition = [3*i_x/4, 3*i_y/4]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
             adaptorSize= adaptorSize-2
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # left, top
             adaptorSize=42+2
             centerPosition = [i_x/4, i_y/4]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
             adaptorSize= adaptorSize-2
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # left, bottom
             centerPosition = [3*i_x/4, i_y/4]
             adaptorSize=18+2
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
             adaptorSize= adaptorSize-2
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray              
        else: # draw test stimuli                
             # right, top
             adaptorSize=42
             centerPosition = [i_x/4, 3*i_y/4]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
             adaptorSize=18
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # right, bottom
             adaptorSize=42
             centerPosition = [3*i_x/4, 3*i_y/4]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
             adaptorSize=18
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # left, top
             adaptorSize=42
             centerPosition = [i_x/4, i_y/4]
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray-testColorChange
             adaptorSize=18
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
             # left, bottom
             centerPosition = [3*i_x/4, i_y/4]
             adaptorSize=42
             startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray-testColorChange
             adaptorSize=18;
             startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray     
       
    if condition==7: # Robinson-deSa 2013
        adaptorSize=42
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors   
          centerPosition = [i_x/2, i_y/2];
          startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
          startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
          # blank out right side
          startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]:centerPosition[1]+adaptorSize/2]=gray
        else: # draw test stimuli
           startinputImage =  np.ones((i_x, i_y))*gray
           centerPosition = [i_x/2, i_y/2]
           startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange   
       
    if condition==8: # Robinson-deSa2012-E1 (stimulus sizes multiply degrees by 10 for pixels)
        # (adaptorSize (100,40,20),testSize (20,100))
        adaptorSize=100
        testSize=100
        centerPosition=[0,0]
        startinputImage = np.zeros((i_x, i_y))
        if time< testOnset:  # draw adaptors   
             startinputImage[:,:]= gray + adaptorColorChange
             # right 
             centerPosition = [i_x/2, 3*i_y/4]
             startinputImage[centerPosition(1)-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]= gray
             # left
             centerPosition = [i_x/2, i_y/4]
             startinputImage[centerPosition(1)-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray
        else: # draw test stimuli
              # equal-sized test stimulus
              startinputImage =  np.ones((i_x, i_y))*gray;
              centerPosition = [i_x/2, i_y/4];
              startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+testColorChange
              # small test stimulus
              centerPosition = [i_x/2, 3*i_y/4];
              startinputImage[centerPosition[0]-round(adaptorSize/4):centerPosition[0]+round(adaptorSize/4), centerPosition[1]-round(adaptorSize/4):centerPosition[1]+round(adaptorSize/4)]=gray+testColorChange    
       
    if condition==9: # Robinson-deSa2012-E2 (stimulus sizes multiply degrees by 10 for pixels)
        # (adaptorSize, middleAdaptorSize, innerAdaptorSize,testSize)
         adaptorSize=100
         middleAdaptorSize=100
         innerAdaptorSize=0
         testSize=100
         centerPosition=[0,0]
         startinputImage = np.zeros((i_x, i_y))
         if time< testOnset:  # draw adaptors   
              startinputImage[:,:]= gray
              # centered 
              centerPosition = [i_x/2, i_y/2]
              # outer edge
              # upper left
              startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0], centerPosition[1]-adaptorSize/2:centerPosition[1]]= gray + adaptorColorChange
              # bottom left
              startinputImage[centerPosition[0]:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]]= gray - adaptorColorChange
              # upper right
              startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0], centerPosition[1]:centerPosition[1]+adaptorSize/2]= gray - adaptorColorChange
              # bottom right
              startinputImage[centerPosition[0]:centerPosition[0]+adaptorSize/2, centerPosition[1]:centerPosition[1]+adaptorSize/2]= gray + adaptorColorChange
              # middle edge
              # upper left
              startinputImage[centerPosition[0]-middleAdaptorSize/2:centerPosition[0], centerPosition[1]-middleAdaptorSize/2:centerPosition[1]]= gray - adaptorColorChange
              # bottom left
              startinputImage[centerPosition[0]:centerPosition[0]+middleAdaptorSize/2, centerPosition[1]-middleAdaptorSize/2:centerPosition[1]]= gray + adaptorColorChange
              # upper right
              startinputImage[centerPosition[0]-middleAdaptorSize/2:centerPosition[0], centerPosition[1]:centerPosition[1]+middleAdaptorSize/2]= gray + adaptorColorChange
              # bottom right
              startinputImage[centerPosition[0]:centerPosition[0]+middleAdaptorSize/2, centerPosition[1]:centerPosition[1]+middleAdaptorSize/2]= gray - adaptorColorChange
              # gray interior
              if innerAdaptorSize>0:
                  startinputImage[centerPosition[0]-innerAdaptorSize/2:centerPosition[0]+innerAdaptorSize/2, centerPosition[1]-innerAdaptorSize/2:centerPosition[1]+innerAdaptorSize/2]= gray
         else: # draw test stimulus
               startinputImage =  np.ones((i_x, i_y))*gray
               # left
               centerPosition = [i_x/2, i_y/2]
               startinputImage[centerPosition[0]-testSize/2:centerPosition[0]+testSize/2, centerPosition[1]-testSize/2:centerPosition[1]+testSize/2]=gray+testColorChange 
            
    if condition==10: # Prediction 
        testSize=50
        centerPosition=[0,0]
        startinputImage = np.ones((i_x, i_y))*gray
        if time< testOnset:  # draw adaptors 
            # illusory contour on left
            for adaptorSize in np.arange(32,5,-8):
                # topright 
                centerPosition = [3*i_x/8, i_y/2-testSize/2 - testSize];
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                # leftright
                centerPosition = [3*i_x/8, i_y/2+testSize/2- testSize];
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
    
                # bottomright 
                centerPosition = [3*i_x/8+testSize, i_y/2-testSize/2- testSize]
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                # bottomleft
                centerPosition = [3*i_x/8+testSize, i_y/2+testSize/2- testSize]
                startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray  
            # middle gray  
            tempinputImage = np.ones(i_x, i_y)*gray
            # copy and paste from temp to startinput
            centerPosition = [i_x/2, i_y/2- testSize]
            startinputImage[centerPosition[0]-testSize/2+1:centerPosition[0]+testSize/2 -1, centerPosition[1]-testSize/2+1:centerPosition[1]+testSize/2-1]=tempinputImage[centerPosition[0]-testSize/2+1:centerPosition[0]+testSize/2 -1, centerPosition[1]-testSize/2+1:centerPosition[1]+testSize/2-1]                   
    
            # drawn contour on right
            for adaptorSize in np.arange(32,5,-8):
                # topright 
                 centerPosition = [3*i_x/8, i_y/2-testSize/2 + testSize]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                 # leftright
                 centerPosition = [3*i_x/8, i_y/2+testSize/2+ testSize]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
     
                 # bottomright 
                 centerPosition = [3*i_x/8+testSize, i_y/2-testSize/2+ testSize]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                 # bottomleft
                 centerPosition = [3*i_x/8+testSize, i_y/2+testSize/2+ testSize]
                 startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]=gray+adaptorColorChange
                 startinputImage[centerPosition[0]-adaptorSize/2+1:centerPosition[0]+adaptorSize/2 -1, centerPosition[1]-adaptorSize/2+1:centerPosition[1]+adaptorSize/2-1]=gray
                 
            # draw line 
            centerPosition = [i_x/2, i_y/2 + testSize]
            startinputImage[centerPosition[0]-testSize/2+1:centerPosition[0]+testSize/2 , centerPosition[1]-testSize/2+1:centerPosition[1]+testSize/2]=gray+adaptorColorChange
            startinputImage[centerPosition[0]-testSize/2+2:centerPosition[0]+testSize/2 -1 , centerPosition[1]-testSize/2+2:centerPosition[1]+testSize/2-1]=gray
           
        else: # draw test stimuli
            startinputImage =  np.ones((i_x, i_y))*gray
            #left gray
            centerPosition = [i_x/2, i_y/2 - testSize]
            startinputImage[centerPosition[0]-testSize/2+1:centerPosition[0]+testSize/2 -1, centerPosition[1]-testSize/2+1:centerPosition[1]+testSize/2-1]=gray+testColorChange
            #right gray
            centerPosition = [i_x/2, i_y/2 + testSize]
            startinputImage[centerPosition[0]-testSize/2+1:centerPosition[0]+testSize/2 -1, centerPosition[1]-testSize/2+1:centerPosition[1]+testSize/2-1]=gray+testColorChange 
    
    # Remove start column and row, add to end, shift matching to Matlab indexing
    #store = np.vstack((startinputImage[1::,:],startinputImage[0,:]))
    #startinputImage = np.vstack((store[:,1::].T,store[:,0])).T
    
    # fixation markers
    startinputImage[i_x/2-2,i_y/2-2]=255
    startinputImage[i_x/2,i_y/2]=255
    startinputImage[i_x/2-2,i_y/2]=0
    startinputImage[i_x/2,i_y/2-2]=0
    
    """
    if makeAnimatedGifs==1:
        filename = "{0}/{1}".format(resultsDirectory,'Stimulus.gif')
        if timeCount==1:
            cv2.imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'loopcount',inf)
        else:
            cv2.imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'writemode','append')
    """
    
    # Convert RGB input image to red-green, blue-yellow, white-black coordinates
    inputImage = np.zeros((i_x, i_y, 3))
    inputImage[:,:,0] = startinputImage
    inputImage[:,:,1] = startinputImage
    inputImage[:,:,2] = startinputImage
    [rg, by, wb] = image_edit.ConvertRGBtoOpponentColor(inputImage, gray)
    
        
    ######################################### LGN CELLS ################################################
    # white black center-surround processing (but with current parameters it actually does very little)
    
    # padding
    PaddingSize = math.floor(np.max(wb.shape)/2) 
    PaddingColor = wb[0,0]
    wb2 = image_edit.im_padding(wb, PaddingSize, PaddingColor)
    
    # convolution
    
    # This method replicates MATLABS conv2 function - but is slow
    OnOff_Excite =  image_edit.conv2(wb2, C, mode='same') # loses imaginary components?
    OnOff_Inhibit = image_edit.conv2(wb2, E, mode='same')
    OffOn_Excite =  image_edit.conv2(wb2, E, mode='same')
    OffOn_Inhibit = image_edit.conv2(wb2, C, mode='same')
    
    # Faster "Raw" Version of Convolution
    #    OnOff_Excite =  np.fft.irfft2(np.fft.rfft2(wb2) * np.fft.rfft2(C, wb2.shape))
    #    OnOff_Inhibit = np.fft.irfft2(np.fft.rfft2(wb2) * np.fft.rfft2(E, wb2.shape))
    #    OffOn_Excite =  np.fft.irfft2(np.fft.rfft2(wb2) * np.fft.rfft2(E, wb2.shape))
    #    OffOn_Inhibit = np.fft.irfft2(np.fft.rfft2(wb2) * np.fft.rfft2(C, wb2.shape))
    
    # shunting parameters
    paramA = 50 # 1 
    paramB = 90 # 90
    paramD = 60 # 60
    
    # shunting
    x_OnOff = (paramB*OnOff_Excite-paramD*OnOff_Inhibit)/(paramA+(OnOff_Excite+OnOff_Inhibit))
    x_OffOn = (paramB*OffOn_Excite-paramD*OffOn_Inhibit)/(paramA+(OffOn_Excite+OffOn_Inhibit))
    
    # cutting negative values
    x_pos = x_OnOff - x_OffOn
    x_pos[x_pos<0] = 0
    x_neg = x_OffOn - x_OnOff
    x_neg[x_neg<0] = 0
    LGNwb = x_pos - x_neg
    
    # pad planes for all color channels for later use
    PaddingColor = wb[0,0]
    wb = image_edit.im_padding(wb, PaddingSize, PaddingColor)
    PaddingColor = rg[0,0]
    rg = image_edit.im_padding(rg, PaddingSize, PaddingColor)
    PaddingColor = by[0,0]
    by = image_edit.im_padding(by, PaddingSize, PaddingColor)
    
    
    ################################ SIMPLE CELL ############################## 
    # Orientations are only based on inputs from the white-black color channel
    
    #    y_pos_1 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNwb) * np.fft.rfft2(F[:,:,0], LGNwb.shape)))
    #    y_pos_2 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNwb) * np.fft.rfft2(F[:,:,1], LGNwb.shape)))
    #    y_pos_3 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNwb) * np.fft.rfft2(F[:,:,2], LGNwb.shape)))
    #    y_pos_4 = np.abs(np.fft.irfft2(np.fft.rfft2(LGNwb) * np.fft.rfft2(F[:,:,3], LGNwb.shape)))
    
    y_pos_1 = np.abs(image_edit.conv2(LGNwb, F[:,:,0], mode='same')) # loses imaginary components?
    y_pos_2 = np.abs(image_edit.conv2(LGNwb, F[:,:,1], mode='same'))
    y_pos_3 = np.abs(image_edit.conv2(LGNwb, F[:,:,2], mode='same'))
    y_pos_4 = np.abs(image_edit.conv2(LGNwb, F[:,:,3], mode='same'))  
    
    y_crop_1 = image_edit.im_cropping(y_pos_1, PaddingSize)
    y_crop_2 = image_edit.im_cropping(y_pos_2, PaddingSize) 
    y_crop_3 = image_edit.im_cropping(y_pos_3, PaddingSize) 
    y_crop_4 = image_edit.im_cropping(y_pos_4, PaddingSize) 
    
    y_crop_1[y_crop_1<0] = 0
    y_crop_2[y_crop_2<0] = 0
    y_crop_3[y_crop_3<0] = 0 
    y_crop_4[y_crop_4<0] = 0
    
    y_crop=np.zeros((y_crop_1.shape[0],y_crop_1.shape[1],K))
    
    y_crop[:,:,0] = y_crop_1
    y_crop[:,:,1] = y_crop_2
    y_crop[:,:,2] = y_crop_3
    y_crop[:,:,3] = y_crop_4
    
    y = y_crop
    
    ############################# COMPLEX CELL ################################ 
    
    # pool across contrast polarity
    planeSize = y.shape
    z1= np.zeros((planeSize[0], planeSize[1], nOrient))
    
    for k in np.arange(0,K/2):
        z1[:,:,k] = y[:,:,k] + y[:,:,k+K/2]
    
    # set upper limit for boundary activity
    boundaryUpperLimit=25;
    z1[z1>boundaryUpperLimit] = boundaryUpperLimit
    
    w1= np.zeros((planeSize[0], planeSize[1], nOrient))
    
    # Add tonic input, inI, to boundaries
    for k in np.arange(0,nOrient):
        w1[:,:,k] = inI  + z1[:,:,k]
    
    
    ############ ORIENTATION SPECIFIC HABITUATING TRANSMITTER GATES ################
    
    #gate = np.zeros(w1.shape)
    # initialize gate on first time step
    if t==startTime :
        gate = Agate/(Bgate + Cgate*inI) * np.ones(w1.shape)
    
    # identify equilibrium solution to gate
    gate_equil = Agate/(Bgate + Cgate* w1)
    
    # solve gate for current time
    gate = gate_equil + (gate - gate_equil)* np.exp(-Rhogate*(Bgate+Cgate*w1)*timeStep)
    
    
    
    ################# CROSS ORIENTATION DIPOLE COMPETITION ##########################
    
    gdAcrossWeight = 0.5 # strength of inhibition between orthogonal orientations
    
    # for k in np.arange(0,nOrient):
    orthgonalK1 = 1 
    orthgonalK2 = 0
    
    v_1 = gate[:,:,0]*w1[:,:,0] - gdAcrossWeight*gate[:,:,orthgonalK1]*w1[:,:,orthgonalK1] 
    v_2 = gate[:,:,1]*w1[:,:,1] - gdAcrossWeight*gate[:,:,orthgonalK2]*w1[:,:,orthgonalK2] 
    """ Error involved here in v_2 unknown origin"""
    
    v_1[v_1<0] = 0  # half-wave rectify
    v_2[v_2<0] = 0  # half-wave rectify
    
    O1=np.zeros((v_1.shape[1],v_1.shape[0],nOrient))
    
    O1[:,:,0] = v_1
    O1[:,:,1] = v_2
    
    # soft threshold for boundaries
    bThresh=9.5
    O2=O1-bThresh
    O2[O2<0] = 0 # These values feed into the filling-in stage
    
    
    ########################## FILLING-IN DOmains [FIDOs] ######################################
    # FIDO  regions of connected boundary points    
    
    
    # Most of this code is an algorithmic way of identifying distinct Filling-In DOmains (FIDOs)
    BndSig = np.sum(O2[:,:,:],2) 
    thint = O2.shape
    BndThr = 0.0
    BndSig = 100*(BndSig - BndThr)
    BndSig[BndSig < 0] = 0
    
    boundaryOrientation = np.zeros((1,nOrient))
    for i in np.arange(0,nOrient):
        boundaryOrientation[0,i] = -np.pi/2 +(i+1)*np.pi/(nOrient)
    
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
        a1=np.abs(np.sin(shiftOrientation[i] - boundaryOrientation[0,0]))
        currentBoundary1[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = a1*(O2[p1[0]:p1[-1]:1,q1[0]:q1[-1]:1,0] + O2[p2[0]:p2[-1]:1,q2[0]:q2[-1]:1,0] )
        
        a2=np.abs(np.sin(shiftOrientation[i] - boundaryOrientation[0,1]))
        currentBoundary2[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = a2*(O2[p1[0]:p1[-1]:1,q1[0]:q1[-1]:1,1] + O2[p2[0]:p2[-1]:1,q2[0]:q2[-1]:1,1] )
        
        currentBoundary=currentBoundary1+currentBoundary2
        a = currentBoundary
        a[a>0] = 1
        a1=dummy[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1]
        a2=    a[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1]
        
        P[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1,i] =   a1- a2
    
    
    # find FIDOs and average within them
    FIDO = np.zeros((sX, sY))
    
    # unique number for each cell in the FIDO
    for i in np.arange(0,sX):
        for j in np.arange(0,sY):
            FIDO[i,j] = i+ j*thint[0]
            
    # Grow each FIDO so end up with distinct domains with a common assigned number
    oldFIDO = np.zeros((sX, sY))
    while np.array_equal(oldFIDO, FIDO) ==0:
        oldFIDO = FIDO;
        for i in np.arange(0,4):
            p = stimarea_x + shift[i,0] 
            q = stimarea_y + shift[i,1] 
            FIDO[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1] = np.maximum(FIDO[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1], FIDO[p[0]:p[-1]:1,q[0]:q[-1]:1]*P[stimarea_x[0]:stimarea_x[-1]:1, stimarea_y[0]:stimarea_y[-1]:1,i])
    
    # input is color signals
    WBColor = image_edit.im_cropping(wb, PaddingSize)
    RGColor = image_edit.im_cropping(rg, PaddingSize)
    BYColor = image_edit.im_cropping(by, PaddingSize)
    
    # Filling-in values for white-black, red-green, and blue-yellow
    S_wb = np.zeros((sX, sY))
    S_rg = np.zeros((sX, sY))
    S_by = np.zeros((sX, sY)) 
    
    # Different filling in methods SLOW PART
    
    ###############################################################
    #uniqueFIDOs = np.unique(FIDO)
    #numFIDOs = uniqueFIDOs.shape  
    #dummyFIDO = np.ones((sX,sY))
    ## Number of pixels in this FIDO
    #for i in np.arange(0,numFIDOs[0]):
    #    Lookup=FIDO==uniqueFIDOs[i]
    #    FIDOsize = np.sum(np.sum(dummyFIDO[Lookup]))
    #    # Get average of color signals for this FIDO
    #    S_wb[Lookup] = np.sum(WBColor[Lookup])/FIDOsize
    #    S_rg[Lookup] = np.sum(RGColor[Lookup])/FIDOsize
    #    S_by[Lookup] = np.sum(BYColor[Lookup])/FIDOsize
    ###############################################################
    #from skimage.segmentation import relabel_sequential
    #
    #WBColor = WBColor.reshape(-1, 3)
    #labels = relabel_sequential(FIDO)[0]
    #labels -= labels.min()
    #
    #def fmeans(double[:, ::1] data, long[::1] labels, long nsp):
    #    cdef long n,  N = labels.shape[0]
    #    cdef int K = data.shape[1]
    #    cdef double[:, ::1] F = np.zeros((nsp, K), np.float64)
    #    cdef int[::1] sizes = np.zeros(nsp, np.int32)
    #    cdef long l, b
    #    cdef double t
    #
    #    for n in range(N):
    #        l = labels[n]
    #        sizes[l] += 1
    #
    #        for z in range(K):
    #            t = data[n, z]
    #            F[l, z] += t
    #
    #    for n in range(nsp):
    #        for z in range(K):
    #            F[n, z] /= sizes[n]
    #
    #return np.asarray(F)
    #
    #color_means = fmeans(WBColor, labels.flatten(), labels.max()+1)
    #
    #mean_image = color_means[labels]
    ###############################################################  
    FIDO = FIDO.astype(int)
    labels = np.arange(FIDO.max()+1, dtype=int)
    S_wb = labeled_mean(WBColor, FIDO, labels)[FIDO]
    S_rg = labeled_mean(RGColor, FIDO, labels)[FIDO]
    S_by = labeled_mean(BYColor, FIDO, labels)[FIDO]
    ###############################################################
    #@jit
    #def numbaloops(Color):
    #    counts=np.zeros(sX*sY)
    #    sums=np.zeros(sX*sY)
    #    S = np.empty((sX, sY))
    #    for x in range(sX):
    #        for y in range(sY):
    #            region=FIDO[x,y]
    #            value=Color[x,y]
    #            counts[region]+=1
    #            sums[region]+=value
    #    for x in range(sX):
    #        for y in range(sY):
    #            region=FIDO[x,y]
    #            S[x,y]=sums[region]/counts[region]
    #    return S 
    #
    #S_wb=numbaloops(WBColor)
    #S_rg=numbaloops(RGColor)
    #S_by=numbaloops(BYColor)
    ###############################################################
    #uniqueFIDOs, unique_counts = np.unique(FIDO, return_counts=True) 
    #numFIDOs = uniqueFIDOs.shape  
    #for i in np.arange(0,numFIDOs[0]):
    #    Lookup = FIDO==uniqueFIDOs[i]
    #    # Get average of color signals for this FIDO
    #    S_wb[Lookup] = np.sum(WBColor[Lookup])/unique_counts[i]
    #    S_rg[Lookup] = np.sum(RGColor[Lookup])/unique_counts[i]
    #    S_by[Lookup] = np.sum(BYColor[Lookup])/unique_counts[i]
    ###############################################################
    # Collections  method of Computing average color for unique FIDOs
    #import collections
    #
    #colors = {'wb': WBColor, 'rg': RGColor, 'by': BYColor}
    #planes = colors.keys()
    #S = {plane: np.zeros((sX, sY)) for plane in planes}
    #
    #for plane in planes:
    #    counts = collections.defaultdict(int)
    #    sums = collections.defaultdict(int)
    #    for (i, j), f in np.ndenumerate(FIDO):
    #        counts[f] += 1
    #        sums[f] += colors[plane][i, j]
    #    for (i, j), f in np.ndenumerate(FIDO):
    #        S[plane][i, j] = sums[f]/counts[f]
    #S_rg=S['rg']
    #S_wb=S['wb']
    #S_by=S['by']
    ###############################################################
    # Pandas method - computing average color for unique FIDOs
    #def newloop():
    #    index=pd.Index(FIDO.flatten(),name='region')
    #    means= pd.DataFrame(RGColor.flatten(),index).groupby(level=0).mean()
    #    lookup=np.zeros(sX*sY)
    #    lookup[means.index]=means.values
    #    return lookup[FIDO]
    #    
    #def newloop1():
    #    index=pd.Index(FIDO.flatten(),name='region')
    #    means= pd.DataFrame(WBColor.flatten(),index).groupby(level=0).mean()
    #    lookup=np.zeros(sX*sY)
    #    lookup[means.index]=means.values
    #    return lookup[FIDO]
    #
    #def newloop2():
    #    index=pd.Index(FIDO.flatten(),name='region')
    #    means= pd.DataFrame(BYColor.flatten(),index).groupby(level=0).mean()
    #    lookup=np.zeros(sX*sY)
    #    lookup[means.index]=means.values
    #    return lookup[FIDO]
    #
    #S_rg=newloop()
    #S_wb=newloop1()
    #S_by=newloop2()
    ################### Save image files of network behavior ######################
    
    # Make boundary animated gif
    orientedImage = np.ones((i_x, i_y,3))  
    step = 1   # can sub-sample image if bigger than 1 
    orientSize = 0
    
    # transform boundary values into intensity signals for viewing image 
    for i in np.arange(0,i_x,step): # -step SPLINT
        for j in np.arange(0,i_y,step): # -step SPLINT
            # if vertical edge at this pixel, color it green
            if O2[i,j,1] >0:
                ratio = O2[i,j,1]/80
                if ratio<0.2:
                    ratio = 0.2
                orientedImage[i, j,2] = 1-ratio # reduce blue -k SPLINT
                orientedImage[i, j,0] = 1-ratio # reduce red  -k SPLINT
    
            # if horizontal edge at this pixel, color it blue
            if O2[i,j,0] >0:
                ratio = O2[i,j,0]/80
                if ratio<0.2:
                    ratio = 0.2
                orientedImage[i, j, 1] = 1-ratio # reduce green
                orientedImage[i, j, 0] = 1-ratio # reduce red
      
    ###########################################################################
    """
    if makeAnimatedGifs==1:
        [imind,cm] = rgb2ind(orientedImage,256)
    
        filename = sprintf('%s/Boundaries.gif', resultsDirectory);
    
        if timeCount==1:
            imwrite(imind,cm,filename,'gif','DelayTime',timeStep,'loopcount',inf);
        else:
            imwrite(imind, cm,filename,'gif','DelayTime',timeStep,'writemode','append');
      
    """
    # Convert values in the color FIDOs to something that can be presented in an image
    S_rgmax = np.max(np.max(np.abs(S_rg[:,:])))
    S_bymax = np.max(np.max(np.abs(S_by[:,:])))
    S_wbmax = np.max(np.max(np.abs(S_wb[:,:])))
    S_max1 = np.maximum(S_rgmax, S_bymax)
    S_max = np.maximum(S_max1, S_wbmax)
    
    # Convert FIDO values to RGB values (relative to gray and maximum FIDO value)
    S_rgb = image_edit.ConvertOpponentColortoRGB(S_rg[:,:], S_by[:,:], S_wb[:,:], gray, S_max)
    # scale to 0 to 255 RGB values
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
    thing = np.ones((i_x, 3*i_y, 3))
    
    # Input image on left (Participant Image)
    thing[:,0:i_y,:]=inputImage/255
    
    # Filled-in values on right (Computer Image)
    thing[:,2*i_y:3*i_y,0]=temp/255 
    thing[:,2*i_y:3*i_y,1]=temp/255  
    thing[:,2*i_y:3*i_y,2]=temp/255
    
    # Boundaries in center (Boundary Image)
    thing[:,i_y:2*i_y,:]=orientedImage # +1 removed from y start SPLINT
    
    # Write individual frame files (with leading zero if less than 10)
    if timeCount>=10:
        filename = "{0}/{1}{2}{3}".format(resultsDirectory, 'All',timeCount,'.png')
    else:
        filename = "{0}/{1}{2}{3}".format(resultsDirectory,'All0',timeCount,'.png')
    
    #Same image to file
    scipy.misc.imsave(filename, thing)
