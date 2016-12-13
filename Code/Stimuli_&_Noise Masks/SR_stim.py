# -*- coding: utf-8 -*-

from whitesillusion import degrees_to_pixels
import numpy as np
from PIL import Image
from make_stimulus import add_blob

def SR_stimuli(bg=(50,170),ppd=30,ssf=5,shape=(10,20),radii = (1.5,0.7),
               square =  60,ellipse =(65,104) ,width =100): 
    
    """ 
    Create stimulus seen in Maertens, Wichman, Shapley 2015
    
    Parameters
    -----------
    bg    - Background value of the stimulus. Default [171cd/m^2 vs 51 cd/m^2]
    ppd   - pixel per degree visual angle
    ssf   - supersampling factor, how much to reduce post-expanded image
    shape - Size of stimuli in degrees of visual angle. (y,x)
    radii - Dimensions square and inner circle (diameter, radius)
    square- Gray values of square
    width - width of gradient between two background luminances
    ellipse - min/max values of inner ellipse luminance (65,104)
    
    Returns
    ----------
    all_stim_sm : Output array is stimuli across range of ellipse luminances
    """    
    
    # Parameters for stimuli 
    ellipse = np.linspace(ellipse[0],ellipse[1],7)

    # create stimulus at 5 times size to allow for supersampling antialiasing
    stim = np.ones(degrees_to_pixels(np.array(shape), ppd).astype(int) * ssf)
    stim[:,0:np.int(stim.shape[1]/2)]=bg[0]
    stim[:,np.int(stim.shape[1]/2):-1]=bg[1]
    
    # intermediate shading between two regions
    for x in np.arange((stim.shape[1]/2)-width,(stim.shape[1]/2)+width,1):
        stim[:,x]=((bg[1]-bg[0])/(2.*width))*x - (stim.shape[1]/2.-width)/2. -140 + bg[0]
    
    # centrepoints of 2 sides
    a=(np.int(stim.shape[1]/4.),np.int(stim.shape[1]*3/4.)) # x co-ordinates 
    b=np.int(stim.shape[0]/2.)                              # y co ordinate 
    
    # draw the two square outer test patches
    radii = degrees_to_pixels(np.array(radii), ppd) * ssf
    radii = radii.astype(np.int64)
    stim[b - radii[0]:b + radii[0], a[0] - radii[0]:a[0] + radii[0]] = square
    stim[b - radii[0]:b + radii[0], a[1] - radii[0]:a[1] + radii[0]] = square
    
    # Draw circles in the centre of the squares
    N=700 # number of points to draw within circle
    all_stim=np.zeros((7,stim.shape[0],stim.shape[1]))
    for i in range(7):
        for t in np.linspace(-np.pi,np.pi,N):
            for r in np.linspace(0,radii[1],N):
                stim[np.int(b + r*np.sin(t)),np.int(a[0] + r*np.cos(t))]=ellipse[i]
                stim[np.int(b + r*np.sin(t)),np.int(a[1] + r*np.cos(t))]=ellipse[i]
        all_stim[i,:,:]=stim
    del stim 
    
    # Shrink the array (supersampling antialiasing)
    all_stim_sm  = np.zeros((7,all_stim.shape[1]/ssf,all_stim.shape[2]/ssf))
    for i in range(7):
        pic = Image.fromarray(np.uint8(all_stim[i,:,:]))
        all_stim_sm[i,:,:]=pic.resize((all_stim.shape[2]/ssf,all_stim.shape[1]/ssf), Image.ANTIALIAS)
    
    del all_stim, pic
    
    return all_stim_sm
    
#stimulus=SR_stimuli()


stim = add_blob(stimulus[2,:,:], positions=(150, 15, increment_value = 0, inc_type = 'ellipse', patch_radius=17)
add_blob(stim, positions = {'x': 150, 'y': 450} , increment_value = 0, inc_type = 'ellipse', patch_radius=17)

# Test plotting 
#import matplotlib.pylab as plt
#plt.imshow(all_stim_sm[2,:,:],cmap='gray',vmin=0,vmax=1)

    