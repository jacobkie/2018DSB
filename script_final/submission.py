#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:36:08 2018

@author: work
"""
import os, glob
import numpy as np
import pandas as pd
from load import load_from_cache, save_to_cache, load_file, load_from_cache_multi


def trans2square(image):
    h, w = image.shape[:2]
    new_shape = max(h, w)
    if (new_shape, new_shape)!=(h, w):
        y1, x1 = (new_shape - h)//2, (new_shape - w)//2
        y2, x2 = new_shape - h-y1, new_shape - w-x1
        image = np.pad(image, ((y1,y2),(x1,x2)), mode='constant', 
                        constant_values = 0)
    else:
        y1, y2, x1, x2 = 0,0,0,0
    return image, y1, y2, x1, x2

def prob_to_rles(lab_img, threshold=10):
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i) 

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def make_submission(preds_df):
    df = load_from_cache('stage2_df')
    result = []
    for ind in preds_df.index:
        mask = preds_df.loc[ind, 'pred']
        assert len(np.unique(mask))==mask.max()+1
        result.append(list(prob_to_rles(mask)))

    new_test_ids=[]
    rles=[]
    
    for n, id_ in enumerate(df['id']):
        rles.extend(result[n])
        new_test_ids.extend([id_]*len(result[n]))
  
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(os.path.join('..//cache','sub-scale3.csv'), index=False)
    


def main():
    ################################################################################
    weight_dir = '/media/work/Data/dsb/cache/UnetRCNN_180410-221747'
    ################################################################################
    df_name = 'stage2_df'
    df = load_from_cache(df_name)
    tags = ['quarter', 'half', None, 'two']
    preds = []
    for tag in tags:
        if tag is None:
            fl = os.path.join(weight_dir, '{}.dat'.format(df_name))
            pred = load_file(fl)
        elif tag == 'two':
            fl_names = glob.glob(os.path.join(weight_dir, '{}_{}'.format(df_name, tag),
                                              '{}_{}_[0-9+].dat'.format(df_name, tag)))+\
                       glob.glob(os.path.join(weight_dir, '{}_{}'.format(df_name, tag), 
                                              '{}_{}_[0-9][0-9].dat'.format(df_name, tag)))
            pred = load_from_cache_multi(os.path.join(weight_dir, 
                                                          '{}_{}'.format(df_name, tag),
                                                          '{}_{}'.format(df_name, tag)),             
                       nb=len(fl_names))
        else:
            fl = os.path.join(weight_dir, '{}_{}.dat'.format(df_name,tag))
            pred = load_file(fl)            
        preds.append(pred)
    
    nb_fls = len(tags)
    results = []
    for ind in df.index:
        masks = [pred.loc[ind, 'pred'] for pred in preds]
        scores = [pred.loc[ind, 'con'] for pred in preds]

        res={}
        for key, vals in zip(np.arange(nb_fls),scores):
            for nb in range(len(vals)):
                res['{}_{}'.format(key, nb)] = vals[nb]                
        res = pd.Series(res).sort_values()
        res = res[res<0.2]
 
        mask = np.zeros_like(masks[0], dtype='int16')
        val = 1
        for ind_res in res.index:
            size, label = ind_res.split('_')
            size, label = int(size), int(label)                        
            index = masks[size]==label+1
            if (np.sum(mask[index]>0)/np.sum(index))<0.5:
                mask[(index)&(mask==0)] = val
                val = val+1
        results.append(mask)
        
    preds_df = pd.DataFrame(index = df.index)
    preds_df['pred'] = results
        
    save_to_cache(preds_df, os.path.join(weight_dir, 'preds_df_scale_01'))
    make_submission(preds_df)
    
if __name__ == "__main__":   
    main()