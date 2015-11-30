# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:28:02 2015

@author: will
"""
import numpy as np
import scipy

def condition1(adaptorSize,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Crosses 
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
    return startinputImage


def condition2(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Blob
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
    return startinputImage


def condition3(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Size match
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
    return startinputImage


def condition4(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Bipartate
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
    return startinputImage


def condition5(centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Pyramid
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
    return startinputImage


def condition6(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange):
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
    return startinputImage


def condition7(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Incomplete
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
    return startinputImage


def condition8(adaptorSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Robinson-deSa 2013 
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
    return startinputImage


def condition9(adaptorSize,testSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Robinson-deSa2012-E1 
    startinputImage = np.zeros((i_x, i_y))
    if time< testOnset:  # draw adaptors   
         startinputImage[:,:]= gray
          # centered 
         centerPosition = [i_x/2, i_y/2]
         startinputImage[centerPosition[0]-adaptorSize/2:centerPosition[0]+adaptorSize/2, centerPosition[1]-adaptorSize/2:centerPosition[1]+adaptorSize/2]= gray + adaptorColorChange
    else: # draw test stimulus
          startinputImage =  np.ones(i_x, i_y)*gray
          # left
          centerPosition = [i_x/2, i_y/2]
          startinputImage[centerPosition[0]-testSize/2:centerPosition[0]+testSize/2, centerPosition[1]-testSize/2:centerPosition[1]+testSize/2]=gray+testColorChange  
    return startinputImage


def condition10(adaptorSize, middleAdaptorSize, innerAdaptorSize,testSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Robinson-deSa2012-E2 
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
    return startinputImage


def condition11(testSize,centerPosition,i_x,i_y,time,testOnset,gray,testColorChange,adaptorColorChange): # Prediciton
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
    return startinputImage
    