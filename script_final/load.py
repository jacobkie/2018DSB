# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:38:13 2017

@author: Jackie
"""
import logging
import numpy as np 
import os, time, pickle, glob
import datetime
from path import Path
import matplotlib.pyplot as plt
import hashlib
import cv2
import pandas  as pd
from PIL import Image
from skimage.segmentation import find_boundaries

logsdout = lambda str: print(time.strftime('%y-%m-%d %H:%M:%S '),'INFO|', str)

def is_gray(image):
    if np.ndim(image)==2:
        return True
    elif image.shape[2]==1:
        return True
    else:
        return np.all((image[:,:,0]==image[:,:,2])&(image[:,:,0]==image[:,:,1]))
    
def plot_image_boundary(df, ind, save_dir, image_col = 'image', mask_col = 'mask',
                        name_id = False, tag=None):
    image = df.loc[ind, image_col]
    image_b = image.copy()
    cval = image.max()
    if 'is_gray' in df.columns:
        gray = df.loc[ind, 'is_gray']
    else:
        gray = is_gray(image)
    mask = df.loc[ind, mask_col]
    if df.loc[ind, 'nb_instance']:    
        mask = mask_12m(mask)
        boundary = np.sum([find_boundaries(mask[:,:,nb], mode='inner') for 
                           nb in range(mask.shape[2])], 0).astype(np.uint8)
    
        image_b[boundary>0]= cval if gray else [cval,0, 0]
    
    id = df.loc[ind, 'id'] if name_id else ind
    id = id if tag is None else '{}_{}'.format(id, tag)
    if 'cut_id' in df.columns:
        name = '{}_{}_{}'.format(id, df.loc[ind, 'cut_id'], df.loc[ind, 'nb_instance'])
    else: 
        name = '{}_{}'.format(id, df.loc[ind, 'nb_instance'])

    plt.imsave(os.path.join(save_dir, name+'_boundary.png'), image_b, 
               cmap=plt.cm.gray if gray else None)
    plt.imsave(os.path.join(save_dir, name+'.png'), image, 
               cmap = plt.cm.gray if gray else None)
    
def read_tif(filepath):
    return np.array(Image.open(filepath))

def read_multi_tif(filepath, n_images = 100):
    img = Image.open(filepath)
    images = []
    for i in range(n_images):
        try:
            img.seek(i)
            images.append(np.array(img))
        except EOFError:
            break
    return np.array(images)

def md5sum(filepath):
    md5 = hashlib.md5()
    image = cv2.imread(filepath)
    md5.update(image)
    return md5.hexdigest()

def mask_m21(gt_mask):
    assert np.ndim(gt_mask)==3
    mask= np.zeros(gt_mask.shape[:2], dtype=np.int32)
    index = gt_mask.max(2)!=0
    mask[index] = gt_mask[index].argmax(1)+1
    return mask

def mask_12m(mask, cval = 1):
    index = mask!=0
    nb_mask = mask[index].max() 
    if nb_mask ==0 :
        return np.zeros(mask.shape + (1,), dtype='uint8')
    else:
        mask_full = np.zeros(mask.shape+(nb_mask,), dtype='uint8') 
        mask_full[index, mask[index]-1]=cval
        return mask_full

def mask_12m_no(mask, vals=None, cval = 1):
    if vals is None:
        vals = np.unique(mask) 
        vals = vals[vals>0]
    index = mask!=0
    nb_mask = len(vals) 
    inds = np.arange(vals.max()+1)
    inds[vals] = np.arange(nb_mask)
    if nb_mask ==0 :
        return np.zeros(mask.shape + (1,), dtype='uint8')
    else:
        mask_full = np.zeros(mask.shape+(nb_mask,), dtype='uint8') 
        mask_full[index, inds[mask[index]]]=cval
        return mask_full
    
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def makefolder(folder, start_time=None, tosecond=False):
    if start_time is None:
        if tosecond:
            start_time = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        else:
            start_time = datetime.datetime.now().strftime('%y%m%d-%H')
    folder = folder + '_{}'.format(start_time)
    if not os.path.exists(folder):
        os.mkdir(folder)
        logsdout('mkdir {}'.format(folder))
    else:
        logsdout('{} already exists'.format(folder))
    return folder

def k2(num, digits=2):
    if num==0:
        return 0
    else:
        if isinstance(num, list):
            return [str(round(x, -int(np.floor(np.log10(abs(x)))-digits)))
                        for x in num]
        else:
            return str(round(num, -int(np.floor(np.log10(abs(num)))-digits)))
def load_file(fl):
    with open(fl, 'rb') as f:
        return pickle.load(f)
    
def save_file(data, fl):
    with open(fl, 'wb') as f:
        pickle.dump(data, f)
        
def load_from_cache(name):
    with open(os.path.join('..','cache','{}.dat'.format(name)),'rb') as f:
        return pickle.load(f)

def save_to_cache(data,name):
    with open(os.path.join('..','cache','{}.dat'.format(name)),'wb') as f:
        pickle.dump(data, f)
        
def save_to_cache_multi(data,name,nb=3,start=0):
    assert isinstance(data, pd.DataFrame)
    size = len(data)//nb+1
    for i in range(nb):
        with open(os.path.join('..','cache','{}_{}.dat'.format(name, start+i)),'wb') as f:
            pickle.dump(data.iloc[(i*size):min((i+1)*size, len(data))], f)        

def load_from_cache_multi2(name, nb=3):
    res = []
    for i in range(nb):
        fl = os.path.join('..','cache','{}_{}'.format(name, i))
        if os.path.exists(fl+'.dat'):
            res.append(load_from_cache('{}_{}'.format(name, i)))
        else:
            fl_names = glob.glob('{}_[0-9].dat'.format(fl))
            res.append(load_from_cache_multi('{}_{}'.format(name, i), len(fl_names)))

    res = pd.concat(res)
    return res

def load_from_cache_multi(name, nb=3):
    res = []
    for i in range(nb):
        with open(os.path.join('..','cache','{}_{}.dat'.format(name, i)),'rb') as f:
            res.append(pickle.load(f))
    res = pd.concat(res)
    return res

class MetricsLogger(object):

    def __init__(self, fname, reinitialize=False):
        self.fname = Path(fname)
        self.reinitialize = reinitialize
        if self.fname.exists():
            if self.reinitialize:
                logging.warn('{} exists, deleting'.format(self.fname))
                self.fname.remove()

    def log(self, message, show_time=True, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        logsdout(message)
        if show_time:
            message = time.strftime('%Y %b %d %H:%M:%S|')+message
        with open(self.fname, 'a') as f:
            f.write(message+'\n')
