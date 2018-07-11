#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:08:32 2018

@author: work
"""
import os, glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from load import save_to_cache, get_ax

def get_df():
    ###################################################################
    data_dir = '/home/work/dsb/ext/TNBC_NucleiSegmentation'
    #correct directory address accordingly
    ######################################################################3
    assert os.path.exists(data_dir)
    image_folders = glob.glob(os.path.join(data_dir, 'Slide*'))
    image_folders = sorted(image_folders)
    df = []
    for image_folder in image_folders:
        image_fls = os.listdir(image_folder)        
        image_fls = sorted(image_fls)
        for image_fl in image_fls:
            filepath = os.path.join(image_folder, image_fl)
            image = cv2.imread(filepath)
            mask_path = filepath.replace('Slide', 'GT')
            mask_unet = cv2.imread(mask_path, 0)
            assert len(np.unique(mask_unet))==2
            _, mask = cv2.connectedComponents(mask_unet, connectivity=4)
            df.append({'image':image, 'mask':mask,
                       'image_path':filepath, 'mask_path': mask_path,
                       'id':image_fl[:-4], 'nb_instance':mask.max(),
                       'shape':image.shape})

    df = pd.DataFrame(df)
    save_to_cache(df, 'TNBC')

if __name__ == "__main__":   
    get_df()