# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 13:09:23 2015

@author: Will
"""
import numpy as np

# Compare Matlab and Python Scripts
import h5py
f = h5py.File('C:\Users\Will\Desktop\workspace_t_whole.mat')
g = h5py.File('C:\Users\Will\Desktop\FIDOinitial.mat')
P22 =  h5py.File('C:\Users\Will\Desktop\P.mat')

comp = g['FIDO']
comp = np.array(comp, dtype='f8')
comp=comp.T
(comp==FIDO).all()

FIDo_initial=comp
LGNwb2diff=(LGNwb2-LGNwb)
Cdiff=C2-C
Ediff=E2-E
np.isclose(E2,E).all()
#FIDO
#C and E

#####################################################################
def Gaussian2D(GCenter, Gamp, Ggamma,Gconst): #new_theta > 0.4:
    new_theta = math.sqrt(Gconst**-1)*Ggamma
    SizeHalf = np.int(math.floor(9*new_theta))
    [y, x] = np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
    part1=(x+GCenter[0])**2+(y+GCenter[1])**2
    GKernel = Gamp*np.exp(-0.5*Ggamma**-2*Gconst*part1)
    return GKernel,SizeHalf,new_theta
    
[a,b,c]=Gaussian2D([0,0],.5,2,np.round(2*np.log(2),4))


a = np.round(2*np.log(2),4)
a = int((a * 10000) + 0.005) / 10000.0 
#####################################################################
# find FIDOs and average within them
FIDO = np.zeros((sX, sY))

# unique number for each cell in the FIDO
for i in np.arange(0,sX):
    for j in np.arange(0,sY):
        FIDO[i,j] = (i+1)+ (j+1)*thint[0]
            
# Grow each FIDO so end up with distinct domains with a common assigned number
# Optimisation process
counter = 0
FIDOstart=FIDO
oldFIDO = 0*FIDO

counter = 0
FIDO=FIDOstart
oldFIDO = 0*FIDO
while (oldFIDO==FIDO).all() == 0: # while arrays not equal
    oldFIDO = FIDO
    counter = counter +1
    for i in np.arange(0,4):
        P_cur = P[:,:,i]
        p = stimarea_x + shift[i,0]
        q = stimarea_y + shift[i,1]
        FIDO[1:199,1:199] = np.maximum(FIDO[1:199, 1:199], FIDO[p[0]:p[-1]+1:1,q[0]:q[-1]+1:1]*P_cur[1:199,1:199])
    
###################################  
sX=sY=200
# find FIDOs and average within them
FIDO_ini = np.zeros((sX, sY))
for i in np.arange(0,sX):
    for j in np.arange(0,sY):
        FIDO_ini[i,j] = (i+1)+ (j+1)*thint[0]  


FIDO_edit=FIDO_ini
oldFIDO=np.zeros((sX,sY))
np.array_equal(oldFIDO, FIDO_edit) == 0
FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1-1:199-1,1:199]*P[1:199, 1:199,0] ) 
FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1+1:199+1,1:199]*P[1:199, 1:199,1] ) 
FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1:199,1-1:199-1]*P[1:199, 1:199,2] ) 
FIDO_edit[1:199, 1:199] = np.maximum(FIDO_edit[1:199, 1:199], FIDO_edit[1:199,1+1:199+1]*P[1:199, 1:199,3] ) 

###############################################################
# SAVE Image
filename3 = "{0}/{1}{2}{3}".format(resultsDirectory,'FIDO_edit',timeCount,'.png')
scipy.misc.imsave(filename3,FIDO_edit)
################################################################
# multiple processing

from multiprocessing import Pool

def multi_conv(A):
    return image_edit.conv2(wb2,A)

pool = Pool(processes=4)
[results1,result2] = pool.map(multi_conv, (C,E,))



from multiprocessing import Pool
    
def multi_conv(A):
    return image_edit.conv2(wb2,A)

pool = Pool(processes=4)
conv_out = pool.map(multi_conv, (C,E,))
    
    
    
import image_edit
from multiprocessing import Process  
import numpy as np
wb2=np.random.rand(200,200)
C=np.random.rand(7,7)
E=C
def multi_conv2(A):
    print image_edit.conv2(wb2,A)
    
p = Process(target=multi_conv2, args=(C,E,))
p.start()
p.join()

##############################################################
import matplotlib.pyplot as plt
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10,10))
ax1.imshow(FIDO_edit[:,:,0])


fig, (ax1) = plt.subplots(ncols=1, figsize=(10,10))
ax1.imshow(FIDO4-FIDO42)

fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(10,10))
ax1.imshow(oldFIDO-FIDO32)
ax2.imshow(P[:,:,0])
ax3.imshow(P2[:,:,0])
#ax3.imshow(FIDO22-oldFIDO)
#ax4.imshow(P[:,:,1])
#ax5.imshow(FIDO32-oldFIDO)
#ax6.imshow(P[:,:,2])
#ax7.imshow(FIDO42-oldFIDO)
#ax8.imshow(P[:,:,3])
#
fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, figsize=(10,10))
ax1.imshow(w1[:,:,0])
ax2.imshow(w12[:,:,0])
ax3.imshow(w1[:,:,1])
ax4.imshow(w12[:,:,1])


fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(20,20))
ax1.imshow(FIDO)#-P2[:,0:5,0])
ax2.imshow(FIDO2)#-P2[:,0:5,1])

fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(ncols=8, figsize=(20,20))
ax1.imshow(P[:,:,0]-P2[:,:,0])#
ax2.imshow(P2[:,:,0])
ax3.imshow(P[:,:,1]-P2[:,:,1])
ax4.imshow(P2[:,:,1])
ax5.imshow(P[:,:,2]-P2[:,:,2])
ax6.imshow(P2[:,:,2])#-P2[:,0:5,2])
ax7.imshow(P[:,:,3]-P2[:,:,3])
ax8.imshow(P2[:,:,3])#-P2[:,0:5,3])

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10,10))
ax1.imshow(temp-temp2)
ax2.imshow(temp2)

#ax5.imshow(y[:,:,0])
#ax6.imshow(y[:,:,1])
#ax7.imshow(y[:,:,2])
#ax8.imshow(y[:,:,3])
#####################################################################
SizeHalf=2
[a,b]=np.meshgrid(np.arange(-SizeHalf,SizeHalf+1), np.arange(-SizeHalf,SizeHalf+1))
#######################################
# Convolution method comparison
from scipy import signal
ypos24=np.abs(signal.convolve(LGNwb2,F2[:,:,0],mode='same'))
ypos2=np.abs(scipy.ndimage.convolve(LGNwb2,F2[:,:,0]))
ypos23=np.abs(image_edit.conv2(LGNwb2, F2[:,:,0], mode='same'))



#####################################################################
# Remove start column and row, add to end, shift matching to Matlab indexing
store = np.vstack((startinputImage[1::,:],startinputImage[0,:]))
startinputImage = np.vstack((store[:,1::].T,store[:,0])).T
#####################################################################
# mess code
import matplotlib.pyplot as plt


#Plotting types
fig, (ax1) = plt.subplots(nrows=1, figsize=(6,10))
ax1.imshow(x_neg)
plt.show()


fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, figsize=(6,10))
ax1.imshow(S_rgb[:,:,0])
ax2.imshow(S_rgb2[:,:,0])
ax3.imshow(S_rgb[:,:,1])
ax4.imshow(S_rgb2[:,:,1])
ax5.imshow(S_rgb[:,:,2])
ax6.imshow(S_rgb2[:,:,2])
plt.show()


############################################################################
""" COMPARE DIFFERENT METHODS' TRUTH """


S_wb1 = np.zeros((sX, sY))
S_rg1 = np.zeros((sX, sY))
S_by1 = np.zeros((sX, sY)) 
# Number of pixels in this FIDO
for i in np.arange(0,numFIDOs[0]):
    FIDOsize = np.sum(np.sum(dummyFIDO[FIDO==uniqueFIDOs[i]]))
    # Get average of color signals for this FIDO
    WBSum = np.sum(np.sum(WBColor[FIDO==uniqueFIDOs[i]]))
    S_wb1[FIDO==uniqueFIDOs[i]] = WBSum/FIDOsize
    RGSum = np.sum(np.sum(RGColor[FIDO==uniqueFIDOs[i]]))
    S_rg1[FIDO==uniqueFIDOs[i]] = RGSum/FIDOsize
    BYSum = np.sum(np.sum(BYColor[FIDO==uniqueFIDOs[i]]))
    S_by1[FIDO==uniqueFIDOs[i]] = BYSum/FIDOsize



S_wb2 = np.zeros((sX, sY))
S_rg2 = np.zeros((sX, sY))
S_by2 = np.zeros((sX, sY)) 
S_wb2=WBColor
S_rg2=RGColor
S_by2=BYColor




###########################################################################

 # Compute average color for unique FIDOs
uniqueFIDOs = np.unique(FIDO)
numFIDOs = uniqueFIDOs.shape  
dummyFIDO = np.ones((sX,sY))
for i in np.arange(0,numFIDOs[0]):
    # Number of pixels in this FIDO (Negate )
    FIDOsize = np.sum(np.sum(dummyFIDO[FIDO==uniqueFIDOs[i]]))
    # Get average of color signals for this FIDO
    WBSum = np.sum(np.sum(WBColor[FIDO==uniqueFIDOs[i]]))
    S_wb[FIDO==uniqueFIDOs[i]] = WBSum/FIDOsize
    RGSum = np.sum(np.sum(RGColor[FIDO==uniqueFIDOs[i]]))
    S_rg[FIDO==uniqueFIDOs[i]] = RGSum/FIDOsize
    BYSum = np.sum(np.sum(BYColor[FIDO==uniqueFIDOs[i]]))
    S_by[FIDO==uniqueFIDOs[i]] = BYSum/FIDOsize
    
    

FIDOsize = np.sum(np.sum(dummyFIDO[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]]))

WBSum = np.sum(np.sum(WBColor[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]]))
S_wb[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]] = WBSum/FIDOsize

RGSum = np.sum(np.sum(RGColor[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]]))
S_rg[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]] = RGSum/FIDOsize

BYSum = np.sum(np.sum(BYColor[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]]))
S_by[FIDO==uniqueFIDOs[0:numFIDOs[0]:1]] = BYSum/FIDOsize
    

##################################################################
# Compute average color for unique FIDOs

S_wb2 = np.zeros((sX, sY))
S_rg2 = np.zeros((sX, sY))
S_by2 = np.zeros((sX, sY)) 

startTeem =  time.clock() 

uniqueFIDOs, unique_counts = np.unique(FIDO, return_counts=True)
numFIDOs = uniqueFIDOs.shape  

for i in np.arange(0,numFIDOs[0]):
    Lookup = FIDO==uniqueFIDOs[i]
    # Get average of color signals for this FIDO
    S_wb2[Lookup] = np.sum(WBColor[Lookup])/unique_counts[i]
    S_rg2[Lookup] = np.sum(RGColor[Lookup])/unique_counts[i]
    S_by2[Lookup] = np.sum(BYColor[Lookup])/unique_counts[i]
    

S_wb = np.zeros((sX, sY))
S_rg = np.zeros((sX, sY))
S_by = np.zeros((sX, sY))
FIDOsize = unique_counts[0:numFIDOs[0]:1]
Lookup = FIDO == uniqueFIDOs[0:numFIDOs[0]:1]
S_wb[0:numFIDOs[0]:1] = np.sum(WBColor[Lookup])/FIDOsize
S_rg[0:numFIDOs[0]:1] = np.sum(RGColor[Lookup])/FIDOsize
S_by[0:numFIDOs[0]:1] = np.sum(BYColor[Lookup])/FIDOsize

###############################################################################
""" Option to make loop run faster"""

import collections
start=time.clock()
colors = {'wb': WBColor, 'rg': RGColor, 'by': BYColor}
planes = colors.keys()
S = {plane: np.zeros((sX, sY)) for plane in planes}

for plane in planes:
    counts = collections.defaultdict(int)
    sums = collections.defaultdict(int)
    for (i, j), f in np.ndenumerate(FIDO):
        counts[f] += 1
        sums[f] += colors[plane][i, j]
    for (i, j), f in np.ndenumerate(FIDO):
        S[plane][i, j] = sums[f]/counts[f]
        
end=time.clock()-start