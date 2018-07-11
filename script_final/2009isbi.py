#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 11:20:22 2018

@author: work
"""
import os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates

from load import get_ax, save_file, md5sum, load_from_cache, save_to_cache
from lib import renumber_mask
from mask_pos import extract_bboxes_4di

def get_images():
    ######################################################################
    data_dir = '/media/work/Ubuntu 16.0/ext/2009_ISBI/data/images/dna-images'
    # where images are stored
    ########################################################################3
    os.path.exists(data_dir)
    df = []
    folders = next(os.walk(data_dir))[1]
    for folder in folders:
        fls = next(os.walk(os.path.join(data_dir, folder)))[2]
        fls = sorted(fls, key = lambda x: int(x[:-4].split('-')[-1]))
        for fl in fls:
            filepath = os.path.join(data_dir, folder, fl)
            image = cv2.imread(filepath)
            df.append({'image':image, 'shape':image.shape, 
                       'md5':md5sum(filepath), 'path':filepath,
                       'id': '{}_{}'.format(folder, fl[:-4])})
    df = pd.DataFrame(df)
    save_to_cache(df, '2009isbi')

def remake_boundary():
    #combine results from paint.net
    #a few nucleus boundaries are not enclosed, which caused my algorithm to fail to
    # include thoese nucleus into the mask. 
    #I used paint.net to fixed this, and reload the corrected boundaries back.
    ##################################################################################
    data_dir = '/home/work/dsb/ext/2009isbi/unclosed_boundary_copy'
    boundary_dir = '/home/work/dsb/ext/2009isbi/segmented-lpc-boundary'
    #data_dir : where manually corrected boundaries are saved
    #boundary_dir: where original boundaries are stored
    ##################################################################################
    fls = os.listdir(data_dir)
    for fl in fls:
        newb = cv2.imread(os.path.join(data_dir, fl), 0)
        colors = np.unique(newb)
        assert len(colors)==2
        color_fg = np.max(colors)
        
        folder, id = fl.split('_')
        bfl = os.path.join(boundary_dir, folder, id)
        boundary = cv2.imread(bfl, 0)
        boundary[newb==color_fg] = np.max(boundary)
        plt.imsave(bfl, boundary, cmap=plt.cm.gray)
        
def get_masks():
    ###############################################################################
    boundary_dir = '/home/work/dsb/ext/2009isbi/segmented-lpc-boundary'
    save_dir = '/home/work/dsb/ext/2009isbi/segmented-lpc-mask'

    ################################################################################
    
    os.path.exists(boundary_dir)    
    folders = next(os.walk(boundary_dir))[1]    
    masks = []
    for folder in folders:
        fls = next(os.walk(os.path.join(boundary_dir, folder)))[2]
        fls = sorted(fls, key = lambda x: int(x[:-4].split('-')[-1]))    
        for fl in fls:
            pass
            id = os.path.basename(fl)[:-4]
            filepath = os.path.join(boundary_dir, folder, fl)
            boundary = cv2.imread(filepath, 0)        
            mask  = boundary2mask(boundary)
            check_sanity(mask, boundary, folder, fl)           
            plt.imsave(os.path.join(save_dir, folder, '{}_mask.png'.format(id)),
                       mask)
            masks.append({'id':'{}_{}'.format(folder, id), 'mask':mask})
 
    masks = pd.DataFrame(masks)    
    df = load_from_cache('2009isbi')
    df = pd.merge(df, masks, how='left', on='id')
    df = df.dropna(0)
    df['nb_instance'] = df['mask'].apply(lambda x: x.max())
    save_to_cache(df, '2009isbi')
    
def fill_boundary(mask, index_boundary):
    y, x = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    y, x = y[index_boundary], x[index_boundary]
    mask[index_boundary] = np.max([map_coordinates(mask, (y+dy,x+dx), order=0, 
        mode='constant') for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),
                        (1,1),(-1,1),(1,-1)]], axis=0)
    return mask
 
def boundary2mask(boundary):
    assert len(np.unique(boundary))==2
    boundary_inverse = boundary.max()-boundary
    _, mask = cv2.connectedComponents(boundary_inverse, connectivity=4)
    vals, counts = np.unique(np.hstack([mask[0], mask[-1], mask[:,0], mask[:,-1]]),
                         return_counts = True)
    bg_label = vals[np.argmax(counts)]
    mask[mask==bg_label]=0
    mask, _ = renumber_mask(mask)
    fill_boundary(mask, boundary>0)
    return mask

def check_sanity(mask, boundary, folder, fl):    
    _, rest = cv2.connectedComponents(((boundary>0)&(mask==0)).astype('uint8'))
    vals, counts = np.unique(rest[rest>0], return_counts=True)
    vals = vals[counts>4]
    counts = counts[counts>4]

    if len(vals):
        plt.imsave(os.path.join('/home/work/dsb/ext/2009isbi/unclosed_boundary',
                            '{}_{}'.format(folder, fl)), (rest>0).astype('uint8')*80)
    return mask

if __name__ == "__main__":   
    get_images()
    remake_boundary() #some boundaries are manually corrected via paint.net and combined
    #with other boundaries
    get_masks()