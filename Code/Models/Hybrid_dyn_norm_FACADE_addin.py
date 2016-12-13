# -*- coding: utf-8 -*-
"""
NOTE: This is not a stand alone piece of code, it is an add in for the Dynamic
Normalisation Network in BEATS.py. Therefore it won't run on it's own.
"""

#""" 
#To lead this into FIDO filling-in 
#first : edge detection then filling in  
#
#"""
#
#import math
#orientationShift=0.5
#gamma=1.75
#PaddingSize = math.floor(c.shape[2]/2) #maximum image dimension /2
#K=4
#G = CANNEM.Gaussian2D([0+orientationShift,0+orientationShift], 1, gamma, 2)
#F = np.zeros((G.shape[0],G.shape[1],K))
#
## ----------- LGN Cell --------------
#for k in np.arange(0,K):
#    m = np.sin((2*np.pi*(k+1))/t_N)
#    n = np.cos((2*np.pi*(k+1))/t_N)
#    H = CANNEM.Gaussian2D([m+orientationShift,n+orientationShift],1,gamma,2)
#    
#    # minus kernels to create ODOG filter for each polarity
#    F[:,:,k] = G - H
#    
#    # normalize positive and negative parts
#    posF = F[:,:,k]
#    (posF>90).choose(posF,0)
#    posnorm = np.sum(np.sum(posF*posF))
#    posF = posF/np.sqrt(posnorm)
#    
#    negF = F[:,:,k]
#    (negF>90).choose(negF,0)
#    negnorm = np.sum(np.sum(negF*negF))
#    negF = negF/np.sqrt(negnorm)
#    
#    # combine positive and negative parts
#    F[:,:,k] = posF + negF
#    
#    # normalize full kernel
#    normalizer = np.sum(np.sum( F[:,:,k]*F[:,:,k] ) )
#    F[:,:,k] = F[:,:,k]/np.sqrt(normalizer)
#
## -----------Simple Cell--------------------
#y = np.zeros((c.shape[1],c.shape[2],K))
#for i in range(K):
#    Ini = np.abs(CANNEM.conv2(c[-2,:,:], F[:,:,i]))   # convolve
#    #Ini = CANNEM.im_cropping(Ini, PaddingSize)         # padding
#    Ini[Ini<0] = 0                                   # half wave rectify
#    y[:,:,i]=Ini
#
## ------------Complex cell-----------------
#z1= np.zeros((y.shape[0], y.shape[1], K/2))
#for k in np.arange(0,K/2):
#    z1[:,:,k] = y[:,:,k] + y[:,:,k+K/2]
#
## Oriented boundary signals thresholded with upper boundary limit
#boundaryUpperLimit=25
#z1[z1>boundaryUpperLimit] = boundaryUpperLimit
#
## Add tonic input, inI, to boundaries
#w1= np.zeros((y.shape[0], y.shape[1], K/2))
#for k in np.arange(0,K/2):
#    w1[:,:,k] = 5  + z1[:,:,k]
#
## ---------- FIDO Filling in --------------------
#
#O2=w1 # ignore dipole component processing for now
#
#shiftOrientation = [np.pi/2, np.pi/2, 0, 0]
#Bshift1 = np.array([[-1, -1],[ 0, -1],[ -1, -1],[ -1, 0]])
#Bshift2 = np.array([[-1,  0],[ 0,  0],[  0, -1],[  0, 0]])
# 
## Prepare boundary signals
#BndSig = np.sum(O2[:,:,:],2) # Sum across orientations
#thint = O2.shape
#BndThr = 0
#BndSig = 100*(BndSig-BndThr)      # Amplify signal
#BndSig[BndSig < 0] = 0            # Half wave rectifier
#
#sX = np.size(BndSig, 0)
#sY = np.size(BndSig, 1)
#stimarea_x = np.arange(1,np.size(BndSig, 0)-1) 
#stimarea_y = np.arange(1,np.size(BndSig, 1)-1)
#
## Setting up boundary structures
#P=np.zeros((sX,sY,4))
#for i in np.arange(0,4):
#    dummy = np.ones((sX, sY))
#    p1 = stimarea_x + Bshift1[i,0]
#    q1 = stimarea_y + Bshift1[i,1]
#    p2 = stimarea_x + Bshift2[i,0]
#    q2 = stimarea_y + Bshift2[i,1]
#    
#    currentBoundary = np.zeros((BndSig.shape[0],BndSig.shape[1]))
#    currentBoundary1 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
#    currentBoundary2 = np.zeros((BndSig.shape[0],BndSig.shape[1]))
#    
#    # for both orientations at each polarity
#    a1=np.abs(np.sin(shiftOrientation[i] - 0))
#    currentBoundary1[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1] = a1*(O2[p1[0]:p1[-1]+1:1,q1[0]:q1[-1]+1:1,0] + O2[p2[0]:p2[-1]+1:1,q2[0]:q2[-1]+1:1,0] )
#    
#    a2=np.abs(np.sin(shiftOrientation[i] - np.pi/2))
#    currentBoundary2[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1] = a2*(O2[p1[0]:p1[-1]+1:1,q1[0]:q1[-1]+1:1,1] + O2[p2[0]:p2[-1]+1:1,q2[0]:q2[-1]+1:1,1] )
#    
#    currentBoundary=currentBoundary1+currentBoundary2
#    a = currentBoundary
#    a[a>0] = 1
#    a1=dummy[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1]
#    a2=    a[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1]
#    
#    P[stimarea_x[0]:stimarea_x[-1]+1:1, stimarea_y[0]:stimarea_y[-1]+1:1,i] =   a1- a2
#        
#
#
## Filling-in stage - find FIDOs and average within them
#
## create unique number for each cell in the FIDO
#FIDO_ini = np.zeros((sX, sY))
#for i in np.arange(0,sX):
#    for j in np.arange(0,sY):
#        FIDO_ini[i,j] = (i+1)+ (j+1)*thint[0]  
#
## Grow each FIDO so end up with distinct domains with a common assigned number
#FIDO_edit=FIDO_ini
#n = 500 # optimization parameter, number of growth steps (Previous while loop with FIDO_edit==FIDO_ini)
#for n in np.arange(1,500):
#    FIDO_edit[0:199, 1:199] = np.maximum(FIDO_edit[0:199, 1:199], np.multiply(FIDO_edit[1-1:199-1,1:199],P[0:199, 1:199,0] ) )
#    FIDO_edit[2:199, 1:199] = np.maximum(FIDO_edit[2:199, 1:199], np.multiply(FIDO_edit[1+1:199+1,1:199],P[2:199, 1:199,1] ) )
#    FIDO_edit[1:199, 0:199] = np.maximum(FIDO_edit[1:199, 0:199], np.multiply(FIDO_edit[1:199,1-1:199-1],P[1:199, 0:199,2] ) )
#    FIDO_edit[1:199, 2:199] = np.maximum(FIDO_edit[1:199, 2:199], np.multiply(FIDO_edit[1:199,1+1:199+1],P[1:199, 2:199,3] ) )
#
#FIDO=FIDO_edit    
#
## input is color signals
#wbColor = c[-2,:,:]#CANNEM.im_cropping(c[-2,:,:], PaddingSize)
#
## Filling-in values for white-black, red-green, and blue-self.yellow
#S_wb = np.zeros((sX, sY))
#
## Compute average color for unique FIDOs
#FIDO = FIDO.astype(int)
#labels = np.arange(FIDO.max()+1, dtype=int)
#S_wb = CANNEM.labeled_mean(wbColor, FIDO, labels)[FIDO]
