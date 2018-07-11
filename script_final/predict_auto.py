#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:41:27 2018

@author: work
"""
import os, glob
from predict_test import predict_test_rcnn_two, predict_test_rcnn, \
                        predict_test_rcnn_half, predict_test_rcnn_quarter, \
                        get_mask, get_mask_scale, get_cons_scale, get_cons
from params import DsbConfig
from load import load_from_cache_multi, save_to_cache_multi, load_from_cache, \
                    save_to_cache

def main():
    config = DsbConfig()
    df_name = 'stage2_df'
    save_name = 'stage2'
    ###################################################################################
    weight_dir = '/home/work/data/dsb/cache/UnetRCNN_180410-221747'
    #correct directory address where model weight for prediction is saved
    ##################################################################################
    predict_test_rcnn(df_name, weight_dir)
    predict_test_rcnn_half(df_name, weight_dir)
    predict_test_rcnn_quarter(df_name, weight_dir)
    predict_test_rcnn_two(df_name, weight_dir)

    #combine predictions for zoom 2
    for nb in range(25):
        fl = os.path.join(weight_dir, 'stage2_df_two_{}.dat'.format(nb))
        if os.path.exists(fl):
            continue
        else:
            fls = glob.glob(os.path.join(weight_dir, 'stage2_df_two_{}_*.dat'.format(nb)))
            preds_df = load_from_cache_multi(fl[:-4], len(fls))
            save_to_cache(preds_df, fl)
            for fl in fls:
                os.remove(fl)
                
                
    get_mask(df_name, config, weight_dir, save_name)
    get_mask_scale(df_name, config, weight_dir, save_name, tag='half')
    get_mask_scale(df_name, config, weight_dir, save_name, tag='quarter')
    
    get_mask_scale(df_name, config, weight_dir, save_name, tag='two')



    get_cons(df_name, weight_dir)
    get_cons_scale(df_name, weight_dir, tag='half')
    get_cons_scale(df_name, weight_dir, tag='quarter')
    get_cons_scale(df_name, weight_dir, tag='two')
if __name__ == '__main__': 
    main()

