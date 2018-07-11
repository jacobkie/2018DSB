#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:25:08 2018

@author: work
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from load import load_from_cache, save_to_cache
from lib import renumber_mask
from params import DsbConfig
from eda import train_valid_split

def get_nx_dx(w, IMAGE_DIM):
    nx = int(np.ceil(w*2/IMAGE_DIM-1))
    dx = int(np.ceil((w - IMAGE_DIM)/(nx-1))) if nx>1 else 0
    return nx, dx

def del_boundary(mask_cut, threshold=20, image_id=None, verbose=0):
    mask_cut = mask_cut.copy()
    mask_values = [np.unique(mask_cut[[0,-1]]),np.unique(mask_cut[:,[0,-1]])]
    mask_values = np.unique(np.hstack(mask_values))
    mask_values = mask_values[mask_values!=0]
    mask_areas = [np.sum(mask_cut == value) for value in mask_values]
    mask_delete = mask_values[np.asarray(mask_areas)<=threshold]
    if len(mask_delete)&verbose:
        print('delete {} masks:{} from image {}'.format(len(mask_delete), 
              mask_delete, image_id))
    for value in mask_delete:
        mask_cut[mask_cut==value]=0
    return mask_cut

def resave_single(image, mask, config, image_id):
    IMAGE_DIM = config.IMAGE_MIN_DIM
    h, w = mask.shape
    nx, dx = get_nx_dx(w, IMAGE_DIM)
    ny, dy = get_nx_dx(h, IMAGE_DIM)
    nb_image = nx*ny
    vals, counts = np.unique(mask[mask!=0], return_counts=True)
    images, masks_original, masks_renumber, image_ids, ids, nb_instance, \
        mask_vals = [], [], [], [], [], [], []

    for nb in range(nb_image):
        nb_x, nb_y = nb%nx, nb//nx
        image_cut = image[dy*nb_y:dy*nb_y+IMAGE_DIM, dx*nb_x:dx*nb_x+IMAGE_DIM].copy()
        mask_cut = mask[dy*nb_y:dy*nb_y+IMAGE_DIM, dx*nb_x:dx*nb_x+IMAGE_DIM].copy()

        mask_cut = del_boundary(mask_cut, threshold=config.MASK_THRESHOLD, 
                                image_id=image_id, verbose=1)

        if mask_cut.shape!=(IMAGE_DIM, IMAGE_DIM):
            padding = ((0, IMAGE_DIM-mask_cut.shape[0]),(0, IMAGE_DIM-mask_cut.shape[1]))
            image_cut = np.pad(image_cut, padding+((0,0),), mode='constant', 
                               constant_values=0)
            mask_cut = np.pad(mask_cut, padding, mode='constant', constant_values=0)
        assert image_cut.shape==(256,256,3), '{}_{} with shape{}'.format(image_id,
                                nb, image_cut.shape)

        mask_re, mask_val = renumber_mask(mask_cut)
        
        images.append(image_cut)
        masks_original.append(mask_cut)
        masks_renumber.append(mask_re)
        mask_vals.append(mask_val)
        image_ids.append(image_id)
        ids.append('{}_{}'.format(nb_image, nb))
        nb_instance.append(mask_re.max())

    return images, masks_original, masks_renumber, image_ids, ids, \
            nb_instance, mask_vals
        
def resave(df, config, n_jobs=8):
    assert config.IMAGE_MIN_DIM == config.IMAGE_MAX_DIM
       
    res = Parallel(n_jobs)(delayed(resave_single)(df.loc[image_id, 'image'], df.loc[image_id, 'mask'],
                                  config, image_id) for image_id in df.index)
    images, masks_original, masks_renumber, image_ids, ids, \
        nb_instance, mask_vals = [],[],[],[],[],[],[]
    for image, mask, mask_re, image_id, id, nb_ins, mask_val in res:
        images += image
        masks_original += mask
        masks_renumber += mask_re 
        image_ids += image_id 
        ids += id 
        nb_instance += nb_ins
        mask_vals += mask_val
    df_cut = pd.DataFrame({'image':images, 'mask_original':masks_original, 
                           'mask':masks_renumber, 'image_id':image_ids, 
                           'cut_id':ids, 'nb_instance':nb_instance,
                           'mask_vals':mask_vals})   
    return df_cut
    
def main():
    config = DsbConfig()
    
    df = load_from_cache('train_df_fixed')
    df_cut = resave(df, config)
    df_cut['shape_id'] = df.loc[df_cut['image_id'].values, 'shape_id'].values
    save_to_cache(df_cut, 'train_df_256_fixed')

    for df_name in ['2009isbi','TNBC', 'weebly']:
        df = load_from_cache(df_name)
        df_cut = resave(df, config)
        save_to_cache(df_cut, '{}_256'.format(df_name))


if __name__ == "__main__":   
    main()
    train_valid_split()