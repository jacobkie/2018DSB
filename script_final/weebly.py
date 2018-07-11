#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:16:42 2018

@author: work
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os

from xml.dom import minidom 
from load import save_to_cache
from lib import renumber_mask
def get_df():
    #########################################################################3
    image_dir = '/home/work/dsb/ext/weebly/Tissue images'
    anno_dir  = '/home/work/dsb/ext/weebly/Annotations'
    #correct directory addresses accordingly here
    ############################################################################
    image_paths = os.listdir(image_dir)
    ids = [x.split('.')[0] for x in image_paths]
    image_paths = [os.path.join(image_dir, path) for path in image_paths]
    mask_paths = [os.path.join(anno_dir, '{}.xml'.format(id)) for id in ids]

    df = []
    for id, image_path, mask_path in zip(ids, image_paths, mask_paths):
        print(id, end=',')
        image = cv2.imread(image_path)
        mask, nb_instance, nb_gt, mask_vals = read_mask(mask_path, image.shape[:2])
        df.append({'image':image, 'mask':mask, 
                   'nb_instance':nb_instance, 'nb_gt': nb_gt,
                   'mask_vals': mask_vals, 'id':id, 
                   'image_path':image_path, 'mask_path':mask_path})    
    df = pd.DataFrame(df)    
    df['shape'] = df['image'].apply(lambda x: x.shape)
    save_to_cache(df, 'weebly') 

def read_mask(mask_path, mask_shape):
    tree = minidom.parse(mask_path)    
    regions = tree.getElementsByTagName("Regions")[0].getElementsByTagName("Region")

    mask = np.zeros(mask_shape, dtype=np.int32)
    for nb, region in enumerate(regions):
        vertices = region.getElementsByTagName("Vertex")
        boundary = []
        for vertex in vertices:
            x = float(vertex.attributes["X"].value)
            y = float(vertex.attributes["Y"].value)
            boundary.append([x,y])
        boundary = np.asarray(boundary).reshape((-1,1,2))
        boundary = np.round(boundary).astype(np.int32)
        cv2.fillPoly(mask, [boundary], nb+1)
    nb_instance = len(np.unique(mask))-1
    nb_gt = len(regions)  #number of regions
    if nb_instance != nb_gt:
        mask, mask_vals = renumber_mask(mask)
    else:
        mask_vals = np.arange(1, nb_gt+1)
    return mask, nb_instance, nb_gt, mask_vals

if __name__ == "__main__":   
    get_df()