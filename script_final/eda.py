#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:16:15 2018

@author: work
"""

import os, glob, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import LabelEncoder

from load import save_to_cache, load_from_cache, logsdout, load_file, mask_m21,\
mask_12m
from params import train_dir, stage2_dir

def is_gray(image):
    if image.shape[2]==1:
        return True
    else:
        return np.all((image[:,:,0]==image[:,:,2])&(image[:,:,0]==image[:,:,1]))
    
def get_train_df(train_dir, save_name):
    train_ids = list(next(os.walk(train_dir)))[1]
    train_df = []
    for id_ in train_ids:
        image = cv2.imread(os.path.join(train_dir, id_, 'images', id_+'.png'))
        gray = is_gray(image)
        shape = image.shape[:2] if gray else image.shape
        masks = []
        for mask_fl in glob.glob(os.path.join(train_dir, id_, 'masks', '*.*')):
            mask_ = cv2.imread(mask_fl, 0)
            masks.append((mask_>0).astype(np.uint8))
        masks = np.stack(masks, 2)
        mask = mask_m21(masks)
        train_df.append({'id':id_, 'image':image, 
                         'is_gray':gray, 'shape':shape, 'mask':mask, 
                         'nb_instance':mask.max()})
    train_df = pd.DataFrame(train_df).sort_values('shape').reset_index()
    train_df['shape_id'] = LabelEncoder().fit_transform(train_df['shape'])
    save_to_cache(train_df, save_name)    

    
def get_test_df(test_dir, save_name='test_df'):
    test_ids = list(next(os.walk(test_dir)))[1]
    test_df = []
    for id_ in test_ids:
        path = os.path.join(test_dir, id_, 'images', id_+'.png')
        image = cv2.imread(path)
        test_df.append({'id':id_, 'image':image, 'path': path, 
                         'shape':image.shape})
    test_df = pd.DataFrame(test_df).sort_values('shape').reset_index()
    save_to_cache(test_df, save_name)    

def train_valid_split():
    df = load_from_cache('train_df_fixed')
    df_256 = load_from_cache('train_df_256_fixed')
    df_256['id'] = df.loc[df_256['image_id'].values, 'id'].values
    train_ids = df['id'][::2]  
    valid_ids = df['id'][1::2]
    train_df = df_256.loc[df_256['id'].isin(train_ids)]
    valid_df = df_256.loc[df_256['id'].isin(valid_ids)]
    save_to_cache(train_df, 'train_final_df')
    save_to_cache(valid_df, 'valid_df')
    
if __name__ == "__main__":   
    get_train_df(train_dir, 'train_df_fixed')
    get_test_df(stage2_dir, 'stage2_df')


