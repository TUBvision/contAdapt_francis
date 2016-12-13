import Image
import matplotlib.pyplot as plt
import numpy as np
import pdb
import unittest
import random
import sys
#import scipy.optimize

__all__ = ['image_to_array', 'array_to_image', 'make_random_array', 'resize_array', 'ellipse', 'smooth_ellipse', 'gauss_ellipse', 'replace_image_region', 'modify_image_region', 'replace_check', 'normalize_image', 'norm_image', 'add_blob', 'make_blob_stimulus', 'euclid_distance']

def image_to_array(fname, in_format = 'png'):
    """
    read specified image file (default: png), converts it to grayscale and into numpy array
    input:
    ------
    fname       - name of image file
    in_format   - extension (png default)
    output:
    -------
    numpy array
    """
    im = Image.open('%s.%s' %(fname, in_format)).convert('L')
    im_matrix = [ im.getpixel(( y, x)) for x in range(im.size[1]) for y in range(im.size[0])]
    im_matrix = np.array(im_matrix).reshape(im.size[1], im.size[0])
    
    return im_matrix


def array_to_image(stimulus_array = None, outfile_name = None, out_format = 'bmp'):
    """
    convert numpy array into image (default = '.bmp') in order to display it with vsg.vsgDrawImage
    input:
    ------
    stimulus_array  -   numpy array
    outfile_name    -   ''
    out_format      -   'bmp' (default) or 'png'
    output:
    -------
    image           -   outfile_name.out_format
    """
    im_row, im_col = stimulus_array.shape
    im_new = Image.new("L",(im_col, im_row))
    im_new.putdata(stimulus_array.flatten())
    im_new.save('%s.%s' %(outfile_name, out_format), format = out_format)


def make_random_array(nr_int=10, min_int=0, max_int=255, side_length=10):
    """
    return a side_length x side_length numpy array consisting of nr_int different values between min_in and max_int that are randomly arranged
    :input:
    --------
    nr_int  - default=10
    min_int - default=0
    max_int - default=255
    side_length - default=10
    
    :output:
    ---------
    numpy array
    """
    total_squares = side_length**2
    n_rep = np.ceil(total_squares/float(nr_int))
    
    intens = np.repeat(np.round(np.linspace(min_int, max_int, nr_int)), n_rep)
    index  = np.arange(len(intens))
    random.shuffle(index)
    index = index[np.arange(total_squares)]
    return np.reshape(intens[index], (side_length, side_length))


def resize_array(arr, factor):
    """
    from Torsten Betz' utils.py
    Return a copy of an array, resized by the given factor. Every value is
    repeated factor[d] times along dimension d.

    Parameters
    ----------
    arr : 2D array
          the array to be resized
    factor : tupel of 2 ints
             the resize factor in the y and x dimensions

    Returns
    -------
    An array of shape (arr.shape[0] * factor[0], arr.shape[1] * factor[1])
    """
    x_idx = np.arange(0, arr.shape[1], 1. / factor[1]).astype(int)
    y_idx = np.arange(0, arr.shape[0], 1. / factor[0]).astype(int)
    return arr[:, x_idx][y_idx, :]


def ellipse(radius, circle=1):
    """
    radius:
    circle: circle==1, ellipse AR < 1: H>V, AR > 1: V>H
    output:
    ---------
    np.array consisting of [1, 0]
    """
    y_size = radius * 2
    x_size = y_size * circle
    x = np.linspace(-1, 1, y_size)
    y = np.linspace(-1, 1, x_size)
    dist = np.fmin(1, np.sqrt(x[np.newaxis, :] ** 2 + y[:, np.newaxis] ** 2))
    d = dist<1
    return d.astype('int')


def smooth_ellipse(radius, circle=1, exponent=2):
    """
    written by Torsten Betz
    Create an ellipse that smoothly falls off from 1 in the center to 0 at
    its edges.

    Parameters
    ----------
    radius : size in pixel of shorter axis
           the shape of the array containing the ellipse. (y,x)
    """
    y_size = radius * 2
    x_size = y_size * circle
    x = np.linspace(-1, 1, y_size)
    y = np.linspace(-1, 1, x_size)
    # compute euclidean distance from center of array for every point, cap at 1.0
    dist = np.fmin(1, np.sqrt(x[np.newaxis, :] ** 2 + y[:, np.newaxis] ** 2))
    # apply cosine to squares of distances, rescaled to have 0 at distance 1 
    return (np.cos(dist ** exponent * np.pi) + 1) / 2


def gauss_ellipse(radius, factor = 2):
    """
    create ellipse with gaussian border profile of width = 2*radius and height = radius
    output:
    ---------
    np.array of floats between [0,1]
    ! as of June, 13, 2012 with adelson_double experiment - gaussian ellipse has double intensity in order to equate summed intensity over ellipse area
    """
    x,y  = np.meshgrid(np.arange(-radius, radius), np.arange(-radius, radius))
    blob = factor * np.exp(- ((( 0.5 * x )**2 + y**2)/(2.0 * radius) ))
    return blob


def raised_cosine(radius, plateau_radius, circle = 1):
    """
    create 2 dimensional cosine with plateau of diameter = 2 * plateau_radius
    :input:
    
    :output:
    """
    x_size = 2 * radius * circle
    y_size = 2 * radius
    x,y  = np.meshgrid(np.linspace(-1, 1, y_size), np.linspace(-1, 1, x_size))
    # clipping of corners - outside unit circle - to 1
    dist = np.fmin(1, np.sqrt(x**2 + y**2))
    
    # set everything within plateau radius to zero
    dist = dist - dist[radius*circle, radius - plateau_radius]
    dist[dist < 0] = 0
    
    # scale image such that maximum is 1
    dist = normalize_image(dist, 0, 1)
    
    return (np.cos(dist * np.pi) + 1) / 2


def generate_grating(bar_width, n_cycles, light, dark, ori = 'v'):
    """
    input:
    ======
    """
    cross = np.array(([dark] * bar_width + [light] * bar_width) * n_cycles)
    if ori == 'h':
        grating = np.outer(cross, np.ones(n_cycles * bar_width * 2))
    else:
        grating = np.outer(np.ones(n_cycles * bar_width * 2), cross)
    return grating


def replace_image_region(orig_image, replace_image):
    """
    replacement of image region
    type: orig_image * x + replace_image * y
    """
    offset = [(orig_image.shape[k] - replace_image.shape[k])/2. for k in [0,1]]
    pos_r  = np.arange(offset[0], offset[0] + replace_image.shape[0])
    pos_c  = np.arange(offset[1], offset[1] + replace_image.shape[1])
    
    new_image = orig_image.copy()
    
    for r, idx_r in enumerate(pos_r):
        for c, idx_c in enumerate(pos_c):
            if replace_image[r,c] != 0:
                new_image[idx_r, idx_c] = orig_image[idx_r, idx_c] * (-(1-0.43) * replace_image[r,c] + 1)  + 85.5 * replace_image[r,c]
    return new_image


def modify_image_region(orig_image, modifier):
    """
    """
    offset = [(orig_image.shape[k] - modifier.shape[k])/2. for k in [0,1]]
    pos_r  = np.arange(offset[0], offset[0] + modifier.shape[0])
    pos_c  = np.arange(offset[1], offset[1] + modifier.shape[1])
    
    new_image = orig_image.copy()
    for r_idx, r in enumerate(pos_r):
        for c_idx, c in enumerate(pos_c):
            new_image[r, c] = orig_image[r, c] + modifier[r_idx, c_idx]
    return new_image


def replace_check(stimulus = None, positions = None, increment_value = 0):
    """
    replaces specified check by central pixel intensity (standard  = default, inc_value = 0), or by central pixel intensity + inc_value (comparison)
    
    keyword arguments: 
    stimulus  - image_array
    positions - dictionary of positions
    value     - increment value
    
    output:
    standard or comparison image array
    (row, col) - coordinates of replaced check
    
    """
    if stimulus == None or positions == None or increment_value is None:
        raise Exception('all keyword arguments are required')
    
    stim = stimulus
    pos  = positions
    value = increment_value
    
    x_asc_start = np.round(np.linspace(pos['x']['up']['start'][0], pos['x']['up']['start'][1], len(pos['y']['up'])))
    x_asc_end   = np.round(np.linspace(pos['x']['up']['end'][0],   pos['x']['up']['end'][1],   len(pos['y']['up'])))
    
    x_dsc_start = np.round(np.linspace(pos['x']['low']['start'][0],pos['x']['low']['start'][1],len(pos['y']['low'])))
    x_dsc_end   = np.round(np.linspace(pos['x']['low']['end'][0],  pos['x']['low']['end'][1],  len(pos['y']['low'])))
    
    x_center, y_center = pos['y']['up'].max(),  pos['x']['low']['start'][1]
    orig_v   = stim[x_center, y_center]
    
    out = stim.copy()
    # upper part of check
    for i, y in enumerate(pos['y']['up']):
        for j in np.arange(x_asc_start[i], x_asc_end[i]):
            out[y, j] = orig_v + value
    
    # lower part of check
    for i, y in enumerate(pos['y']['low']):
        for j in np.arange(x_dsc_start[i], x_dsc_end[i]):
            out[y, j] = orig_v + value
    
    return out.round(), (x_center, y_center)


def normalize_image(stim_in, new_min = 1, new_max = 256):
    """
    input:
    ----------
    stim_in     - numpy array
    new_min
    new_max
    scale image range from [old_min, old_max] to [new_min=1, new_max = 256]
    """
    stim = stim_in.copy()
    stim = stim - stim.min()
    stim = stim/float(stim.max())
    stim = stim * (new_max - new_min)
    stim = stim + new_min
    if stim.max() > 1:
        stim.round()
    return stim


def norm_image(fname='../../stimuli/csgAdelson45', bg=120):
    """
    read image, normalize to range [min, max] and set background
    input
    =====
    fname - adelson checkerboard
    background
    
    output
    ======
    image numpy array
    """
    stim_in = image_to_array(fname)
    stim    = normalize_image(stim_in, 1,256)
    stim[stim == 128] = bg
    return stim


def add_blob(stimulus = None, positions = None, increment_value = 0, inc_type = 'ellipse', patch_radius=17):
    """
    add gaussian patch of aspect_ratio 0.5 and peak intensity of increment value to specified check
    
    keyword arguments:
    stimulus  - image_array
    positions - dictionary of positions
    value     - increment value
    inc_type  - ellipse, smooth_ellipse, cosine_ellipse, circle, smooth_circle, cosine_circle
    output:
    standard or comparison image array
    (row, col) - coordinates of replaced check
    """
    if stimulus == None or positions == None or increment_value is None:
        raise Exception('all keyword arguments are required')
    
    #patch_radius = 17
    #patch_radius = positions['y']['low'].max() - positions['y']['low'].min()
    
    if inc_type == 'ellipse':
        blob = ellipse(patch_radius, 0.5)
    
    elif inc_type == 'smooth_ellipse':
        patch_radius = np.int(1.3 * patch_radius)
        blob         = smooth_ellipse(patch_radius, 0.5) * 1.14
    
    elif inc_type == 'xsmooth_ellipse':
        patch_radius = np.int(1.56 * patch_radius)
        blob         = smooth_ellipse(patch_radius, 0.5, 1) * 1.36
    
    elif inc_type == 'cosine_ellipse':
        patch_radius = np.int(1.65 * patch_radius)
        blob         = raised_cosine(patch_radius, 1/7. * patch_radius, 0.5)
    
    elif inc_type == 'circle':
        blob = ellipse(patch_radius)
    
    elif inc_type == 'smooth_circle':
        patch_radius = np.int(1.2 * patch_radius)
        blob         = smooth_ellipse(patch_radius) * 1.44
    
    elif inc_type == 'xsmooth_circle':
        patch_radius = np.int(1.56 * patch_radius)
        blob         = smooth_ellipse(patch_radius, 1, 1) * 1.36
    
    elif inc_type == 'cosine_circle':
        patch_radius = np.int(1.65 * patch_radius)
        blob         = raised_cosine(patch_radius, 1/7.*patch_radius, 1) * 1.36
    
    #print 'increment: %d, blob_max: %d' %(increment_value, blob.max())
    
    y_center = 150#positions['y']
    x_center = 150#positions['x']
    #positions['y']['up'].max(),  positions['x']['low']['start'][1]
    patch_y, patch_x = blob.shape
    x_pos  = np.arange(x_center - np.int(patch_x/2.), x_center - np.int(patch_x/2.)+patch_x)
    y_pos  = np.arange(y_center - np.int(patch_y/2.), y_center - np.int(patch_y/2.)+patch_y)
    
    # homogenous patch
    out, dum = replace_check(stimulus, positions,  0)
    
    for i, x in enumerate(x_pos):
        for j, y in enumerate(y_pos):
            out[y,x] = np.round(out[y,x] + blob[j,i] * increment_value)
    
    return out.round(), (x_center, y_center)


def make_blob_stimulus(standard_blob='out_dark', comparison_blob='in', standard_inc=20, comparison_inc=20, background=120, inc_type='ellipse', inc_radius=17):
    """
    input:
    ------
    inctype         -   'check', 'blob'
    patch_position  -   'in', 'out_dark', 'out_bright'
    increment_value -   0 (= default, homogenize) - max_increment
    background      -   background intensity (default: 120)
    inc_type        -   one of 'ellipse', 'smooth_ellipse', 'xsmooth_ellipse', 'circle', 'smooth_circle', xsmooth_circle
    inc_radius      -   in checkerboard default 17, 
    output:
    -------
    comp            -   new image
    coord           -   center coordinate tuple of manipulated check
    """
    
    # dictionary with check positions
    # ===============================
    position = {}
    position['out_dark']   = {'x': {'up': {'start': [217, 177], 'end':[220, 258]}, 'low':{'start':[177, 217], 'end':[259, 218]}}, 'y': {'up': np.arange(245, 262), 'low': np.arange(262, 280)}}
    position['out_bright'] = {'x': {'up': {'start': [173, 134], 'end':[179, 216]}, 'low':{'start': [133, 172], 'end':[216, 175]}}, 'y':  {'up': np.arange(264, 281), 'low': np.arange(280, 298)}}
    position['in'] = {'x': {'up': {'start': [259, 219], 'end':[262, 303]}, 'low':{'start':[218, 260], 'end':[302, 261]}}, 'y': {'up': np.arange(299, 318), 'low': np.arange(317, 337)}}
    
    stim = norm_image(bg=background)
    comp, coord = add_blob(stimulus=stim, positions=position[standard_blob],   increment_value=standard_inc,   inc_type=inc_type, patch_radius=inc_radius)
    comp, coord = add_blob(stimulus=comp, positions=position[comparison_blob], increment_value=comparison_inc, inc_type=inc_type, patch_radius=inc_radius)
    
    print 'standard: %s: %d,  comparison: %s: %d ' %(standard_blob, standard_inc, comparison_blob, comparison_inc)
    return comp, coord


def euclid_distance(p1, p2):
    """
    compute euclidean distance between 2 points in same space
    input:
    -------
    p1, p2  -   tuples of equal dimension
    output:
    -------
    distance
    """
    
    if len(p1) != len(p2):
        raise ValueError('p1 and p2 must be of equal dimension')
    
    return np.sqrt(np.sum([(p1[p] - p2[p])**2 for p in np.arange(len(p1))]))


def gen_stim(flank_width=200, gradient_width=120, gradient_direction=1, d=40, lum_range=(50,100), left_outer=75, right_outer=75):
    """
    :Arguments:
    ----------
    flank_width     200
    gradient_width  100
    gradient_direction 1
    d               120
    lum_range       (50,100)
    left_outer      75
    right_outer     75
    :Output:
    ----------
    """
    min_lum, max_lum = lum_range
    
    # gradient direction
    if gradient_direction == 1:
        left_lum, right_lum = min_lum, max_lum
    else:
        left_lum, right_lum = max_lum, min_lum
    
    width  = 2 * flank_width + gradient_width
    height = np.int(width/2.)
    
    # generate empty array and gradient
    im = np.zeros((height, width))
    gradient = np.round(np.linspace(left_lum, right_lum, gradient_width))
    
    for l in range(height):
        # fill left part
        for m in range(flank_width):
            im[l, m] = left_lum
        # fill middle part
        for k, n in enumerate(range(flank_width, flank_width+gradient_width)):
            im[l, n] = gradient[k]
        # fill right part
        for o in range(flank_width+gradient_width, width):
            im[l, o] = right_lum
    
    # generate equiluminant discs
    mean_lum = int((left_lum+right_lum) / 2.)
    #disc     = make_stimulus.ellipse(d) * left_outer
    disc     = np.ones((d*2,d*2)) * left_outer
    d_h, d_w = disc.shape
    
    radius  = d
    
    # add left disc to stimulus
    x1 = flank_width/2 - radius
    x2 = flank_width/2 + radius
    y1 = height/2 - radius
    y2 = height/2 + radius
    
    for k, r in enumerate(range(x1, x2)):
        for l, c in enumerate(range(y1, y2)):
            if disc[k, l] == 0:
                im[c, r] = im[c, r]
            else:
                im[c, r] = disc[k, l]
    
    # add right disc to stimulus
    x1 = flank_width + gradient_width + flank_width/2 - radius
    x2 = flank_width + gradient_width + flank_width/2 + radius
    y1 = height/2 - radius
    y2 = height/2 + radius
    
    for k, r in enumerate(range(x1, x2)):
        for l, c in enumerate(range(y1, y2)):
            if disc[k, l] == 0:
                im[c, r] = im[c, r]
            else:
                im[c, r] = disc[k, l]
    
    return im


def add_increment(stimulus=None, increment=None, position=None):
    """
    :Input:
    ----------
    stimulus    - numpy array of original stimulus
    increment   - numpy array of to be added increment
    position    - tuple of center coordinates within stimulus where increment should be placed
    :Output:
    ----------
    """
    
    inc_y, inc_x  = increment.shape
    pos_y, pos_x  = position
    
    x1 = pos_x - inc_x/2
    x2 = pos_x + inc_x/2
    y1 = pos_y - inc_y/2
    y2 = pos_y + inc_y/2
    
    for k, c in enumerate(range(x1, x2)):
        for l, r in enumerate(range(y1, y2)):
            stimulus[r, c] = stimulus[r, c] + increment[l, k]
    return stimulus


def replace_image_part(stimulus=None, replacement=None, position=None):
    """
    :Input:
    ----------
    stimulus    - numpy array of original stimulus
    increment   - numpy array of to be added increment
    position    - tuple of center coordinates within stimulus where increment should be placed
    :Output:
    ----------
    """
    
    inc_y, inc_x  = replacement.shape
    pos_y, pos_x  = position
    
    x1 = pos_x - inc_x/2
    x2 = pos_x + inc_x/2
    y1 = pos_y - inc_y/2
    y2 = pos_y + inc_y/2
    
    new_stimulus = stimulus.copy()
    
    for k, c in enumerate(range(x1, x2)):
        for l, r in enumerate(range(y1, y2)):
            new_stimulus[r, c] = replacement[l, k]
    return new_stimulus



class TestAnalysisTools(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_png_to_array(self):
        pass
        #try:
            #standard = Image.open('test.png').convert('L')
        #except:
            #raise Exception('test image not found')
    
    def test_euclid_distance(self):
        d = euclid_distance((2,1), (5,5))
        self.assertAlmostEqual(d, 5.0, places = 10)
        self.assertRaises(ValueError, euclid_distance, (2,1), (5,5,2))


