# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:26:40 2015
Translated Python Version of Gregory Francis' contour adaptation
"""

# From the commnand line, go to the directory with the images and type
# convert -quality 100 -dither none -delay 10 -loop 0 All*.png Movie.gif

import numpy as np
import math
import Gaussian2D


# Code to make GIFs or make PNGS in file
makeAnimatedGifs=0 # 0- make PNGs, 1- make GIFs

# Parameters
gray = 127

###############Define kernels and other paramters ##############

# Kernels for LGN (Lateral geniculate nucleus?)

# excitatory condition
C_ = 10 # coefficient (C>E) 18
alpha = .5# radius of spreads (a<b) .5

# inhibitory condition
E_ = .5 # .5
beta = 2 # 2

def Gaussian2D(GCenter, Gamp, Ggamma,Gconst):
    new_theta = np.sqrt(Gconst**-1)*Ggamma
    if new_theta < .4:
        print('kernel is too small!')
    SizeHalf = int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(range(-SizeHalf,SizeHalf+1), range(-SizeHalf,SizeHalf+1))
    part1=(x+GCenter[0])**2+(y+GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)    
    return GKernel

C = np.round(Gaussian2D([0,0], C_, alpha, round(2*np.log(2),3)),3)
E = np.round(Gaussian2D([0,0], E_, beta, 2*np.log(2)),3)

# Parameters for Orientation GDs
Agate = 20.0  
Bgate = 1.0
Cgate = 1.0
Rhogate = 0.007  
inI = 5 

# number of orientations for simple cells
K = 4  # number of polarities (4 means horizontal and vertical orientations)
nOrient = K/2  
orientationShift = 0.5  # boundaries are shifted relative to image plane (boundaries are between pixels)

gamma = 1.75
G = Gaussian2D([0+orientationShift, 0+orientationShift], 1, gamma, 2)
F = np.zeros([G.shape[0],G.shape[1],4])

# Orientation filters (Difference of Offset Gaussians)
for k in np.arange(0,K):
    m = np.sin((2*np.pi*k)/K)
    n = np.cos((2*np.pi*k)/K)
    H = Gaussian2D([m+orientationShift, n+orientationShift], 1, gamma, 2)
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


# Filling-in stage paramters  
# Boundaries are shifted in a grid plane relative to color/brightness
# signals

shift = np.array([[-1,0],[1,0],[0,-1],[0, 1]]) # up, down, left, right
shiftOrientation = [np.pi/2, np.pi/2, 0, 0] # orientation of flow

# Boundaries that block flow for the corresponding shift
Bshift1 = np.array([[-1, -1],[ 0, -1],[ -1, -1],[ -1, 0]])
Bshift2 = np.array([[-1,  0],[ 0,  0],[  0, -1],[  0, 0]])

# image size
i_x=200
i_y=200
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
# set up folder for results for this run
Conditions = {'Crosses', 'Blob', 'SizeMatch', 'Bipartite', 'Pyramids', 'Annuli', 'Incomplete', 'Robinson-deSa2013', 'Robinson-deSa2012-E1', 'Robinson-deSa2012-E2', 'Prediction'}

for condition in np.range(1,11):
    fprintf('Simulating condition %s. \n', char(Conditions(condition)))
    resultsDirectory = sprintf('Results/Condition%s', char(Conditions(condition)))
    if exist(resultsDirectory)==0:
        mkdir(resultsDirectory)

    timeStep = 0.1 # seconds
    stopTime = 6 # seconds 6 for A&G conditions - makes for 4 second adaptation
    startTime = timeStep
    testOnset = stopTime-2.0 # seconds
    testColorChange=5 # 5 is typical for A&G conditions (this is the change, increase or decrease, from the gray background)

    if condition==8:  # special settings for R&dS 2013 condition
        timeStep = 0.153 # seconds
        stopTime = 8.12 # seconds -- 6.12 seconds of adaptation
    #	stopTime = 2+timeStep # seconds - no adapt condition (actually one presentation) -- 0.153 seconds of adaptation
        startTime = timeStep
        testOnset = stopTime-2.0 # seconds
        testColorChange = 8  # 8 is good for demonstration

    if condition==9 or condition==10: # special settings for R&dS 2012 conditions
        timeStep = 0.16 # seconds
        stopTime = 7.0 # seconds -- 5 seconds of adaptation (R&dS 2012 had long initial period and then short re-adapt)
    #	stopTime = 2+3*timeStep; % seconds -no adapt condition (actually one flicker cycle of adaptation)
        startTime = timeStep
        testOnset = stopTime-2.0 # seconds
        testColorChange = 8  # 8 is good for demonstration of same size and larger size adaptor  

    timeCount=0
    for time in np.arange(startTime,stopTime,timeStep):
        timeCount= timeCount+1
    ########################### INPUT STIMULUS ############################### 
        # make input all gray
        startinputImage = 0*np.zeros(i_x, i_y) + gray
        # dhange adaptor color with each time step to produce flicker
        adaptorColorChange = -gray # black
        if mod(timeCount, 2)== 0:
            adaptorColorChange=gray
        if condition==1: # Crosses
            adaptorSize = 42; # divisible by 2 and by 6
            for i in np.arange(1,4+1):	 
                 centerPosition = [0, 0]
                 if i==1: # left
                    centerPosition = [i_x/2, i_y/4]
                 if i==2: # right
                    centerPosition = [i_x/2, 3*i_y/4]
                 if i==3: # top
                    centerPosition = [1*i_x/4, i_y/2]
                 if i==4: # bottom
                    centerPosition = [3*i_x/4, i_y/2]   
                 centerPosition(1) = np.round(centerPosition(1))
                 centerPosition(2) = np.round(centerPosition(2))
                 if time< testOnset: # draw adaptors   
                     # draw crosses
                     if(i==3 or i==4):
                         # vertical
                         startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/6:centerPosition(2)+adaptorSize/6]=gray+adaptorColorChange
                         # horizontal
                         startinputImage[centerPosition(1)-adaptorSize/6:centerPosition(1)+adaptorSize/6, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                         # make outline, by cutting middle
                         startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/6+1:centerPosition(2)+adaptorSize/6-1]=gray
                         startinputImage[centerPosition(1)-adaptorSize/6+1:centerPosition(1)+adaptorSize/6 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                 else: # draw test stimuli
                     testColor = gray+testColorChange
                     if i==1 or i==3:
                         testColor=gray-testColorChange
                     # vertical
                     startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/6:centerPosition(2)+adaptorSize/6]=testColor
                     # horizontal
                     startinputImage[centerPosition(1)-adaptorSize/6:centerPosition(1)+adaptorSize/6, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=testColor   
        # end Crosses

        if condition==2: # Blob 
            adaptorSize = 42 # divisible by 2 and by 6
            startinputImage = 0*np.zeros(i_x, i_y)
            if time< testOnset:  # draw adaptors   
                # right (blurry square)
                centerPosition = [i_x/2, 3*i_y/4]
                startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]= adaptorColorChange
                # blur entire image before putting other elements on
                temp = ones(10)/100;
                [Blur] = conv2(startinputImage, temp, 'same');
                startinputImage = Blur + gray;  
                # left
                centerPosition = [i_x/2, i_y/4];
                startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
            else: # draw test stimuli
               startinputImage =  0*np.zeros(i_x, i_y)+gray;
               centerPosition = [i_x/2, i_y/4];
               startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
               centerPosition = [i_x/2, 3*i_y/4];
               startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange    
          # end Blob

        if condition==3: # SizeMatch 
             adaptorSize = 42 # divisible by 2 and by 6
             startinputImage = 0*np.zeros(i_x, i_y)+gray;
             if time< testOnset:  # draw adaptors   
                  # left
                  adaptorSize = 54
                  centerPosition = [round(2*i_x/3), i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # right
                  adaptorSize = 30
                  centerPosition = [round(2*i_x/3),3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # top
                  adaptorSize = 42
                  centerPosition = [i_x/4,2*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray  
             else: # draw test stimuli
                   # left
                   centerPosition = [round(2*i_x/3), i_y/4]
                   startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
                    # right
                   centerPosition = [round(2*i_x/3), 3*i_y/4]
                   startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
                    # top
                   centerPosition = [i_x/4, i_y/2]
                   startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
        # end SizeMatch

        if condition==4: # Bipartite 
             adaptorSize = 120 # divisible by 2 and by 6
             startinputImage = 0*zeros(i_x, i_y)+gray
             if time< testOnset:  # draw adaptors   
                  # center vertical line
                  centerPosition = [i_x/2, i_y/2]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2):centerPosition(2)]=gray+adaptorColorChange;              
             else:
                  # draw test stimuli
                  # darker gray for background
                  startinputImage = 0*np.zeros(i_x, i_y)+gray-30; 
                  # left side
                  centerPosition = [i_x/2, i_y/2]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)]=gray+testColorChange                  
                   # right side
                  centerPosition = [i_x/2, i_y/2]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2):centerPosition(2)+adaptorSize/2]=gray                      
        # end Bipartite

        if condition==5: # Pyramids 
             startinputImage = 0*zeros(i_x, i_y) + gray
             if time< testOnset:  # draw adaptors   
                  for adaptorSize in np.arange(62,10,-16):
                      # right 
                      centerPosition = [i_x/2, 3*i_y/4]
                      startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                      startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                      # left
                      centerPosition = [i_x/2, i_y/4]
                      startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                      startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
             else: # draw test stimuli
                  startinputImage =  0*np.zeros(i_x, i_y)+gray
                  # draw pyramids
                  count=1
                  for adaptorSize in np.arange(62,10,-16):
                       centerPosition = [i_x/2, i_y/4];
                       startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray-testColorChange*count
                       centerPosition = [i_x/2, 3*i_y/4];
                       startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange*count    
                       count=count+1

                  count=count-1;
                  # draw comparison centers
                  centerPosition = [i_x/4, i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray-testColorChange*count
                  centerPosition = [i_x/4, 3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange*count                
        # end Pyramids      

        if condition==6: # Annuli 
             startinputImage = 0*np.zeros(i_x, i_y) + gray
             if time< testOnset:  # draw adaptors  (lines are thicker than for other demos) 
                  # right, top
                  centerPosition = [i_x/4, 3*i_y/4]
                  adaptorSize=42+2
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  adaptorSize= adaptorSize-2
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # right, bottom
                  adaptorSize=18+2
                  centerPosition = [3*i_x/4, 3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  adaptorSize= adaptorSize-2
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # left, top
                  adaptorSize=42+2
                  centerPosition = [i_x/4, i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  adaptorSize= adaptorSize-2
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # left, bottom
                  centerPosition = [3*i_x/4, i_y/4]
                  adaptorSize=18+2
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  adaptorSize= adaptorSize-2
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray              
             else: # draw test stimuli                
                  # right, top
                  adaptorSize=42
                  centerPosition = [i_x/4, 3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
                  adaptorSize=18
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # right, bottom
                  adaptorSize=42
                  centerPosition = [3*i_x/4, 3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
                  adaptorSize=18
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # left, top
                  adaptorSize=42
                  centerPosition = [i_x/4, i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray-testColorChange
                  adaptorSize=18
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # left, bottom
                  centerPosition = [3*i_x/4, i_y/4]
                  adaptorSize=42
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray-testColorChange
                  adaptorSize=18;
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray              
        # end Annuli      

        if condition==7: # Incomplete 
                adaptorSize = 84 # divisible by 2 and by 6
                startinputImage = 0*np.zeros(i_x, i_y)+ gray;
                if time< testOnset:  # draw adaptors   
                  centerPosition = [i_x/2, i_y/2];
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                  startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                  # blank out right side
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2):centerPosition(2)+adaptorSize/2]=gray
                else: # draw test stimuli
                   startinputImage =  0*np.zeros(i_x, i_y)+gray;
                   centerPosition = [i_x/2, i_y/2];
                   startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange 
        # end Incomplete   

        if condition==8: # Robinson-deSa 2013
             adaptorSize = 42; # divisible by 2 and by 6
             startinputImage = 0*np.zeros(i_x, i_y)
             if time< testOnset:  # draw adaptors   
                  startinputImage[:,:]= gray + adaptorColorChange
                  # right 
                  centerPosition = [i_x/2, 3*i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]= gray
                  # left
                  centerPosition = [i_x/2, i_y/4]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray
             else: # draw test stimuli
                   # equal-sized test stimulus
                   startinputImage =  0*zeros(i_x, i_y)+gray;
                   centerPosition = [i_x/2, i_y/4];
                   startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+testColorChange
                   # small test stimulus
                   centerPosition = [i_x/2, 3*i_y/4];
                   startinputImage[centerPosition(1)-round(adaptorSize/4):centerPosition(1)+round(adaptorSize/4), centerPosition(2)-round(adaptorSize/4):centerPosition(2)+round(adaptorSize/4)]=gray+testColorChange    
        # end Robinson-deSa 2013 

        if condition==9: # Robinson-deSa2012-E1 (stimulus sizes multiply degrees by 10 for pixels)
             adaptorSize = 100 # Large - 100, medium - 40, small - 20
             testSize = 100  # small 20, large 100
             startinputImage = 0*np.zeros(i_x, i_y)
             if time< testOnset:  # draw adaptors   
                  startinputImage[:,:]= gray
                   # centered 
                  centerPosition = [i_x/2, i_y/2]
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]= gray + adaptorColorChange
             else: # draw test stimulus
                   startinputImage =  0*np.zeros(i_x, i_y)+gray
                   # left
                   centerPosition = [i_x/2, i_y/2]
                   startinputImage[centerPosition(1)-testSize/2:centerPosition(1)+testSize/2, centerPosition(2)-testSize/2:centerPosition(2)+testSize/2]=gray+testColorChange
        # end Robinson-deSa2012-E1           

        if condition==10: # Robinson-deSa2012-E2 (stimulus sizes multiply degrees by 10 for pixels)
             adaptorSize = 140
             middleAdaptorSize=100 #always matches testSize
             innerAdaptorSize=0 # size of inner gray square
             testSize = 100  
             startinputImage = 0*np.zeros(i_x, i_y)
             if time< testOnset:  # draw adaptors   
                  startinputImage[:,:]= gray
                  # centered 
                  centerPosition = [i_x/2, i_y/2]
                  # outer edge
                  # upper left
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1), centerPosition(2)-adaptorSize/2:centerPosition(2)]= gray + adaptorColorChange
                  # bottom left
                  startinputImage[centerPosition(1):centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)]= gray - adaptorColorChange
                  # upper right
                  startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1), centerPosition(2):centerPosition(2)+adaptorSize/2]= gray - adaptorColorChange
                  # bottom right
                  startinputImage[centerPosition(1):centerPosition(1)+adaptorSize/2, centerPosition(2):centerPosition(2)+adaptorSize/2]= gray + adaptorColorChange
                  # middle edge
                  # upper left
                  startinputImage[centerPosition(1)-middleAdaptorSize/2:centerPosition(1), centerPosition(2)-middleAdaptorSize/2:centerPosition(2)]= gray - adaptorColorChange
                  # bottom left
                  startinputImag[centerPosition(1):centerPosition(1)+middleAdaptorSize/2, centerPosition(2)-middleAdaptorSize/2:centerPosition(2)]= gray + adaptorColorChange
                  # upper right
                  startinputImage[centerPosition(1)-middleAdaptorSize/2:centerPosition(1), centerPosition(2):centerPosition(2)+middleAdaptorSize/2]= gray + adaptorColorChange
                  # bottom right
                  startinputImage[centerPosition(1):centerPosition(1)+middleAdaptorSize/2, centerPosition(2):centerPosition(2)+middleAdaptorSize/2]= gray - adaptorColorChange
                  # gray interior
                  if innerAdaptorSize>0:
                      startinputImage[centerPosition(1)-innerAdaptorSize/2:centerPosition(1)+innerAdaptorSize/2, centerPosition(2)-innerAdaptorSize/2:centerPosition(2)+innerAdaptorSize/2]= gray
             else: # draw test stimulus
                   startinputImage =  0*np.zeros(i_x, i_y)+gray
                   # left
                   centerPosition = [i_x/2, i_y/2]
                   startinputImage[centerPosition(1)-testSize/2:centerPosition(1)+testSize/2, centerPosition(2)-testSize/2:centerPosition(2)+testSize/2]=gray+testColorChange 
        # end Robinson-deSa2012-E2  

        if condition==11: # Prediction 
                startinputImage = 0*np.zeros(i_x, i_y) + gray
                testSize = 50
                if time< testOnset:  # draw adaptors 
                    
                    # illusory contour on left
                    for adaptorSize in np.arange(32,5,-8):
                        # topright 
                        centerPosition = [3*i_x/8, i_y/2-testSize/2 - testSize];
                        startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                        startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                        # leftright
                        centerPosition = [3*i_x/8, i_y/2+testSize/2- testSize];
                        startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                        startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray

                        # bottomright 
                        centerPosition = [3*i_x/8+testSize, i_y/2-testSize/2- testSize]
                        startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                        startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                        # bottomleft
                        centerPosition = [3*i_x/8+testSize, i_y/2+testSize/2- testSize]
                        startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                        startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray  
                    # middle gray  
                    tempinputImage = 0*np.zeros(i_x, i_y) + gray
                    # copy and paste from temp to startinput
                    centerPosition = [i_x/2, i_y/2- testSize]
                    startinputImage[centerPosition(1)-testSize/2+1:centerPosition(1)+testSize/2 -1, centerPosition(2)-testSize/2+1:centerPosition(2)+testSize/2-1]=tempinputImage[centerPosition(1)-testSize/2+1:centerPosition(1)+testSize/2 -1, centerPosition(2)-testSize/2+1:centerPosition(2)+testSize/2-1]                   

                    # drawn contour on right
                    for adaptorSize in np.arange(32,5,-8):
                        # topright 
                         centerPosition = [3*i_x/8, i_y/2-testSize/2 + testSize]
                         startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                         startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                         # leftright
                         centerPosition = [3*i_x/8, i_y/2+testSize/2+ testSize]
                         startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                         startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
 
                         # bottomright 
                         centerPosition = [3*i_x/8+testSize, i_y/2-testSize/2+ testSize]
                         startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                         startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                         # bottomleft
                         centerPosition = [3*i_x/8+testSize, i_y/2+testSize/2+ testSize]
                         startinputImage[centerPosition(1)-adaptorSize/2:centerPosition(1)+adaptorSize/2, centerPosition(2)-adaptorSize/2:centerPosition(2)+adaptorSize/2]=gray+adaptorColorChange
                         startinputImage[centerPosition(1)-adaptorSize/2+1:centerPosition(1)+adaptorSize/2 -1, centerPosition(2)-adaptorSize/2+1:centerPosition(2)+adaptorSize/2-1]=gray
                    end
                    # draw line 
                    centerPosition = [i_x/2, i_y/2 + testSize]
                    startinputImage[centerPosition(1)-testSize/2+1:centerPosition(1)+testSize/2 , centerPosition(2)-testSize/2+1:centerPosition(2)+testSize/2]=gray+adaptorColorChange
                    startinputImage[centerPosition(1)-testSize/2+2:centerPosition(1)+testSize/2 -1 , centerPosition(2)-testSize/2+2:centerPosition(2)+testSize/2-1]=gray
                   
                else: # draw test stimuli
                    startinputImage =  0*zeros(i_x, i_y)+gray
                    #left gray
                    centerPosition = [i_x/2, i_y/2 - testSize]
                    startinputImage[centerPosition(1)-testSize/2+1:centerPosition(1)+testSize/2 -1, centerPosition(2)-testSize/2+1:centerPosition(2)+testSize/2-1]=gray+testColorChange
                    #right gray
                    centerPosition = [i_x/2, i_y/2 + testSize]
                    startinputImage[centerPosition(1)-testSize/2+1:centerPosition(1)+testSize/2 -1, centerPosition(2)-testSize/2+1:centerPosition(2)+testSize/2-1]=gray+testColorChange 
         # end Prediction            
          
          
        # fixation markers
        startinputImage(i_x/2-1,i_y/2-1)=255
        startinputImage(i_x/2+1,i_y/2+1)=255
        startinputImage(i_x/2-1,i_y/2+1)=0
        startinputImage(i_x/2+1,i_y/2-1)=0
        
        if makeAnimatedGifs==1:
            filename = sprintf('%s/Stimulus.gif', resultsDirectory)
            if timeCount==1:
                imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'loopcount',inf)
            else:
                imwrite(startinputImage,filename,'gif','DelayTime',timeStep,'writemode','append')
               
        # Convert RGB input image to red-green, blue-yellow, white-black
        # coordinates
        inputImage = np.zeros(i_x, i_y, 3)
        inputImage[:,:,1] = startinputImage
        inputImage[:,:,2] = startinputImage
        inputImage[:,:,3] = startinputImage
        [rg, by, wb] = ConvertRGBtoOpponentColor(inputImage, gray)
        
        #################### LGN CELLS ###########################
        # white black center-surround processing (but with current
        # parameters it actually does very little)
        # padding
        PaddingSize = math.floor(max(size(wb))/2)  # /3
        PaddingColor = wb(1,1)
        wb2 = im_padding(wb, PaddingSize, PaddingColor)

        # convolution
        [OnOff_Excite] = conv2(wb2, C, 'same')
        [OnOff_Inhibit] = conv2(wb2, E, 'same')

        [OffOn_Excite] = conv2(wb2, E, 'same')
        [OffOn_Inhibit] = conv2(wb2, C, 'same')

        # shunting parameters
        paramA = 50; # 1 
        paramB = 90; # 90
        paramD = 60; # 60

        # shunting
        x_OnOff = (paramB*OnOff_Excite-paramD*OnOff_Inhibit)/(paramA+(OnOff_Excite+OnOff_Inhibit))
        x_OffOn = (paramB*OffOn_Excite-paramD*OffOn_Inhibit)/(paramA+(OffOn_Excite+OffOn_Inhibit))

        # cutting negative values
        x_pos = x_OnOff - x_OffOn; x_pos(x_pos<0) = 0
        x_neg = x_OffOn - x_OnOff; x_neg(x_neg<0) = 0

        LGNwb = x_pos - x_neg

        # pad planes for all color channels for later use
        PaddingColor = wb(1,1)
        wb = im_padding(wb, PaddingSize, PaddingColor)
        PaddingColor = rg(1,1)
        rg = im_padding(rg, PaddingSize, PaddingColor)
        PaddingColor = by(1,1)
        by = im_padding(by, PaddingSize, PaddingColor)

        ########################### SIMPLE CELL ########################## 
        # Orientations are only based on inputs from the white-black color
        # channel
        for k in np.range(1,K+1):
            # convolution
            y_pos_ = abs(conv2(LGNwb, F[:,:,k], 'same'))     
            y_crop_ = im_cropping(y_pos_, PaddingSize)	
            y_crop_(y_crop_<0) = 0
            y_crop[:,:,k] = y_crop_ 

        y = y_crop

        ######################### COMPLEX CELL ########################### 

        # pool across contrast polarity
        planeSize = size(y)
        z1= np.zeros(planeSize[1], planeSize[2], nOrient)
        
        for k in np.arange(1,(K/2) + 1):
            z_ = y[:,:,k] + y[:,:,k+K/2]
            z1[:,:,k] = z_

        # set upper limit for boundary activity
        boundaryUpperLimit=25;
        z1(z1>boundaryUpperLimit) = boundaryUpperLimit

        w1= np.zeros(planeSize(1), planeSize(2), nOrient);

        # Add tonic input, inI, to boundaries
        for k in np.arange(1,nOrient+1):
            w1[:,:,k] = inI  + z1[:,:,k]

        ############ Habituating gates for each orientation #############

        # initialize gate on first time step
        if time==startTime :
            gate = Agate/(Bgate + Cgate*inI) * ones(size(w1))

        # identify equilibrium solution to gate
        gate_equil = Agate/(Bgate + Cgate* w1)
        # solve gate for current time
        gate = gate_equil + (gate - gate_equil)* exp(-Rhogate*(Bgate+Cgate*w1)*timeStep)
        
        ########### Dipole Competition Between Perpendicular Orientations #######
        
        gdAcrossWeight = 0.5; # strength of inhibition between orthogonal orientations
        for k in np.arange(1,nOrient+1):
            orthgonalK = mod(k+nOrient/2-1,nOrient)+1  # compute index of orthogonal orientation
            v_ = gate[:,:,k]*w1[:,:,k] - gdAcrossWeight*gate[:,:,orthgonalK]*w1[:,:,orthgonalK] 
            v_(v_<0) = 0;  # half-wave rectify
            O1[:,:,k] = v_;

        # soft threshold for boundaries
        bThresh=9.5
        O2=O1-bThresh
        O2(O2<0) = 0 # These values feed into the filling-in stage

        ########################### FILLING-IN ############################## 
        """clear P"""

        # Most of this code is an algorithmic way of identifying distinct
        # Filling-In DOmains (FIDOs)
        BndSig = np.sum(O2[:,:,:,1],3)
        thint = np.size(O2)
        BndThr = 0.0
        BndSig = 100*(BndSig - BndThr)
        BndSig(BndSig < 0) = 0

        boundaryOrientation = np.zeros(1,nOrient) 
        for i in np.arange(1,nOrient+1):
            boundaryOrientation(1,i) = -pi/2 +i*pi/nOrient;
        
        sX = np.size(BndSig, 1)
        sY = np.size(BndSig, 2)
        P_sum = mp.zeros(sX, sY)
        stimarea_x = np.arange(2,np.size(BndSig, 1)) 
        stimarea_y = np.arange(2,np.size(BndSig, 2))

        # Setting up boundary structures
        for i in np.arange(1,4+1):
            P_cur = np.ones(sX, sY)
            dummy = np.ones(sX, sY)
            p1 = stimarea_x + Bshift1(i,1)
            q1 = stimarea_y + Bshift1(i,2)
            p2 = stimarea_x + Bshift2(i,1)
            q2 = stimarea_y + Bshift2(i,2)
            currentBoundary = np.zeros(BndSig.shape )
            for k  in np.arange(1,nOrient+1):
                currentBoundary(stimarea_x, stimarea_y) = currentBoundary(stimarea_x, stimarea_y) + abs(sin(shiftOrientation(i) - boundaryOrientation(1,k)))*(O2(p1, q1,k) + O2(p2, q2,k) );

            temp = currentBoundary
            temp(temp>0) = 1
            P_cur(stimarea_x, stimarea_y) = dummy(stimarea_x, stimarea_y)  - temp(stimarea_x, stimarea_y)
            P_sum = P_sum + P_cur
            P[i] = P_cur

        # find FIDOs and average within them
        FIDO = np.zeros(sX, sY )

        # unique number for each cell in the FIDO
        for i in np.arange(1,sX+1):
            for j in np.arange(1,sY+1):
                FIDO(i,j) = i+ j*thint(1)
        # Grow each FIDO so end up with distinct domains with a common
        # assigned number
        oldFIDO = 0*FIDO
        while np.array_equal(oldFIDO, FIDO) ==0:
            oldFIDO = FIDO;
            for i in np.arange( 1,4+1):
                P_cur = P[i]
                p = stimarea_x + shift[i,1]
                q = stimarea_y + shift[i,2]
                FIDO[stimarea_x, stimarea_y] = np.max(FIDO(stimarea_x, stimarea_y), FIDO(p,q)*P_cur(stimarea_x, stimarea_y) )

        #  compute average color signal for each unique FIDO
        uniqueFIDOs = unique(FIDO) # set of indices that correspond to a FIDO
        numFIDOs = size(uniqueFIDOs)       
        dummyFIDO = np.ones(sX,sY)

        # input is color signals
        WBColor =  im_cropping(wb, PaddingSize)
        RGColor = im_cropping(rg, PaddingSize)
        BYColor = im_cropping(by, PaddingSize)

        # Filling-in values for white-black, red-green, and blue-yellow
        S_wb = np.zeros(sX, sY) 
        S_rg = np.zeros(sX, sY) 
        S_by = np.zeros(sX, sY) 

        # Compute average white-black, red-green, and blue-yellow for all
        # pixels in a common FIDO
        for i in np.arange(1,numFIDOs(1)+1):
            # Number of pixels in this FIDO
            FIDOsize = sum(sum(dummyFIDO(FIDO==uniqueFIDOs(i))))
            # Get average of color signals for this FIDO
            WBSum = sum(sum(WBColor(FIDO==uniqueFIDOs(i))))
            S_wb(FIDO==uniqueFIDOs(i)) = WBSum/FIDOsiz
            RGSum = sum(sum(RGColor(FIDO==uniqueFIDOs(i))))
            S_rg(FIDO==uniqueFIDOs(i)) = RGSum/FIDOsize
            BYSum = sum(sum(BYColor(FIDO==uniqueFIDOs(i))))
            S_by(FIDO==uniqueFIDOs(i)) = BYSum/FIDOsize

        ################# Save image files of network behavior ##############
        
        # Make boundary animated gif
        orientedImage = ones(i_x, i_y,3)  
        step = 1   # can sub-sample image if bigger than 1 
        orientSize = 0

        # transform boundary values into intensity signals that will show
        # up well in an image (this transformation is appropriate for the
        # current simulations, but may not generalize for other
        # simulations)
        for i in np.arange(step,i_x,step):
            for j in np.arange(step,i_y,step):
            # if vertical edge at this pixel, color it green
                if O2(i,j,2) >0:
                    ratio = O2(i,j,2)/80;
                if ratio<0.2:
                  ratio = 0.2;

                orientedImage[i+k, j,3] = 1-ratio # reduce blue 
                orientedImage[i+k, j,1] = 1-ratio # reduce red

                # if horizontal edge at this pixel, color it blue
                if O2(i,j,1) >0:
                    ratio = O2(i,j,1)/80;
                if ratio<0.2:
                    ratio = 0.2;

                orientedImage[i, j, 2] = 1-ratio # reduce green
                orientedImage[i, j,1] = 1-ratio # reduce red
                
        if makeAnimatedGifs==1:
            [imind,cm] = rgb2ind(orientedImage,256)

            filename = sprintf('%s/Boundaries.gif', resultsDirectory);

            if timeCount==1:
                imwrite(imind,cm,filename,'gif','DelayTime',timeStep,'loopcount',inf);
            else:
                imwrite(imind, cm,filename,'gif','DelayTime',timeStep,'writemode','append');
      
        
        # Convert values in the color FIDOs to something that can be
        # presented in an image
        
        S_rgmax = np.max(np.max(np.abs(S_rg[:,:])))
        S_bymax = np.max(np.max(np.abs(S_by[:,:])))
        S_wbmax = np.max(np.max(np.abs(S_wb[:,:])))
        S_max1 = np.max(S_rgmax, S_bymax)
        S_max = np.max(S_max1, S_wbmax)

        # Convert FIDO values to RGB values (relative to gray and maximum
        # FIDO value)
        S_rgb = ConvertOpponentColortoRGB(S_rg[:,:], S_by[:,:], S_wb[:,:], gray, S_max)
        # scale to 0 to 255 RGB values
        temp = 255.0* (S_rgb[:,:,1]/np.max(np.max(np.max(S_rgb))))

        if makeAnimatedGifs==1:
            filename = sprintf('%s/FilledIn.gif', resultsDirectory)

            if timeCount==1:
                imwrite(temp,filename,'gif','DelayTime',timeStep,'loopcount',inf)
            else:
                imwrite(temp,filename,'gif','DelayTime',timeStep,'writemode','append')

        # Make image of input, boundaries, and filled-in values to save as
        # a png file
        thing = ones(i_x, 3*i_y, 3)

        # Input image on left
        thing[:,1:i_y,:]=inputImage/255

        # Filled-in values on right
        thing[:,2*i_y+1:3*i_y,1]=temp/255
        thing[:,2*i_y+1:3*i_y,2]=temp/255    
        thing[:,2*i_y+1:3*i_y,3]=temp/255

        # Boundaries in center
        thing[:,i_y+1:2*i_y,:]=orientedImage

        #Write individual frame files (with leading zero if less than 10)
        if timeCount>=10:
            filename = sprintf('%s/All%d.png', resultsDirectory, timeCount);
        else:
            filename = sprintf('%s/All0%d.png', resultsDirectory, timeCount);

        [im, map] = rgb2ind(thing, 256)    
        imwrite(thing, filename, 'png')
        
        # time loop

# End of condition loop