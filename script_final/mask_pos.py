#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 00:27:52 2018

@author: work
"""
import numpy as np
import cv2
import time
from cv2 import GaussianBlur
from skimage.segmentation import find_boundaries
import pandas as pd

from load import load_from_cache,save_to_cache, get_ax, k2
from params import DsbConfig
from joblib import Parallel, delayed

def main():
    config = DsbConfig()
    df = load_from_cache('train_df_256_clean')  
    labels0 = get_labels(config.IMAGE_SHAPE[0])
    coords = get_coords(config.IMAGE_SHAPE[0])

    for nb in np.arange(0,4000,50):
        mask = df.loc[nb, 'mask']
        start = time.time()
        t0 = mask_localpos(mask, 'all')
        time0 = time.time()-start
        t = mask_localpos_new(mask, 'all', labels0, coords)
        time1 = time.time()-start-time0
        print(nb, np.allclose(t, t0), 'time_ratio:{}'.format(k2(time0/time1,4)))
 
    nb = 800
    mask_local = mask_localpos(mask, tp='all')
    ax = get_ax(2, 3, 8)

    ax[0,0].imshow(df.loc[nb,'image'])
    ax[0,1].imshow(df.loc[nb,'mask'])
    ax[0,2].imshow(mask_local[:,:,0])
    ax[1,0].imshow(mask_local[:,:,1])
    ax[1,1].imshow(mask_local[:,:,2])
    ax[1,2].imshow(mask_local[:,:,3])

def get_labels(img_size):
    labels = np.zeros((img_size, img_size, 4), dtype='int32')
    labels[:,:,1] = np.arange(img_size*img_size).reshape((img_size, img_size))    
    labels[:,:,0] = labels[:,:,1].T
    cumsum = np.cumsum(np.hstack([0, np.arange(1, img_size+1),np.arange(img_size-1,0,-1)]))
    for i in range(2*img_size-1):
        ys = np.arange(i+1) if i<img_size else np.arange(i+1-img_size, img_size)
        labels[ys, i-ys, 3] = np.arange(cumsum[i], cumsum[i+1])
        labels[ys, ys+(img_size-1-i), 2] = np.arange(cumsum[i], cumsum[i+1])
    
    labels0 = np.ones((img_size+2, img_size+2, 4), dtype='float32')*-5
    labels0[1:-1,1:-1] = labels
    labels0[0,:,0]=labels0[1,:,0]-1/3
    labels0[-1,:,0]=labels0[-2,:,0]+1/3
    labels0[:,0,1]=labels0[:,1,1]-1/3
    labels0[:,-1,1]=labels0[:,-2,1]+1/3
    labels0[0,:-2,2]=labels0[1,1:-1,2]-1/3
    labels0[:-2,0,2]=labels0[1:-1,1,2]-1/3
    labels0[-1,2:,2]=labels0[-2,1:-1,2]+1/3
    labels0[2:,-1,2]=labels0[1:-1,-2,2]+1/3
    labels0[0,2:,3] =labels0[1,1:-1,3]-1/3 
    labels0[:-2,-1,3]=labels0[1:-1,-2,3]-1/3
    labels0[-1,:-2,3]=labels0[-2,1:-1,3]+1/3
    labels0[2:,0,3] =labels0[1:-1,1,3]+1/3
    return labels0

def get_coords(img_size):
    labels = get_labels(img_size)[1:-1,1:-1]
    h, w = img_size, img_size
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.astype('float32'), y.astype('float32')
    dr,dl = x+y, y-x    
    coords = np.zeros((img_size*img_size, 4), dtype='float32')
    for nb, val in enumerate([y,x,dr,dl]):
        df = pd.Series(val.flatten(), index = labels[:,:,nb].flatten())
        coords[:,nb]=df.sort_index().values
    return coords

def extract_bboxes_4di(mask, tp='all'):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    nb_mask = mask.max()
    if nb_mask==0:
        return []
    else:
        h, w = mask.shape
        boxes = np.zeros([nb_mask, 8], dtype=np.int32)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        index = mask!=0   
        x, y = x[index], y[index]
        m = mask[index]
        for i in range(nb_mask):
            # Bounding box.
            index_mask = m==i+1
            horizontal_indicies = x[index_mask]
            vertical_indicies = y[index_mask]
            diagonal_right = (horizontal_indicies+vertical_indicies)
            diagonal_left = (vertical_indicies-horizontal_indicies)
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies.min(), horizontal_indicies.max()
                y1, y2 = vertical_indicies.min(), vertical_indicies.max()
                dr1,dr2= diagonal_right.min(), diagonal_right.max()
                dl1,dl2= diagonal_left.min(), diagonal_left.max()    
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2, dr1, dr2, dl1, dl2 = 0, 0, 0, 0, 0, 0, 0, 0
            boxes[i] = np.array([y1, y2, x1, x2, dr1, dr2, dl1, dl2])
        boxes = boxes.astype(np.int32)
        if tp=='all':
            return boxes 
        elif tp=='xy':
            return boxes[:, :4]
        elif tp=='di':
            return boxes[:, 4:]

def mask_relpos(mask, tp='all'):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    nb_mask = mask.max()
    nb_feature = 4 if tp=='all' else 2
    nb_start = 2 if tp=='di' else 0
    if nb_mask==0:
        return np.zeros(mask.shape+(nb_feature,), dtype='float32')
    else:
        bbox = extract_bboxes_4di(mask, tp)    
        assert nb_mask == len(bbox)

        h, w = mask.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.astype('float32'), y.astype('float32')
        x[mask==0]=0
        y[mask==0]=0
        dr,dl = x+y, y-x

        res = [y, x, dr, dl][nb_start:nb_start+nb_feature]
        for i in range(nb_mask):
            # Bounding box.
            index_mask = mask==i+1
            for nb_box in range(nb_feature):
                delta = bbox[i, 2*nb_box+1]-bbox[i, 2*nb_box]
                if delta!=0:
                    res[nb_box][index_mask] = 2*(res[nb_box][index_mask]-
                       bbox[i, 2*nb_box])/delta-1
                else:
                    res[nb_box][index_mask] = 1e-7
        return np.stack(res, axis=2)


def extract_local_mask(mask, tp='all'):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    bbox = extract_bboxes_4di(mask, tp)
    nb_mask = mask.max()
    img_size = len(mask)
    assert nb_mask == len(bbox) 
    nb_feature = 4 if tp=='all' else 2
    nb_start = 2 if tp=='di' else 0
    
    local_mask = np.zeros(mask.shape+(8,), dtype='int32')
    for i in range(nb_mask):
        y1, y2, x1, x2, r1, r2, l1, l2 = bbox[i]
        index_mask = mask==i+1
        
        for px in range(x1, x2+1):
            vals = index_mask[:, px]
            local_mask[vals,px,:2]=get_exts(vals)

        for py in range(y1, y2+1):
            vals = index_mask[py]
            local_mask[py,vals,2:4]=get_exts(vals)

        for pl in range(l1, l2+1):
            if pl<=0:
                ys = np.arange(img_size+pl)
            else:
                ys = np.arange(pl, img_size)
            vals = index_mask[ys, ys-pl]
            local_mask[ys[vals],ys[vals]-pl,4:6]=get_exts(vals, startpoint=abs(pl), spacing=2)
            
        for pr in range(r1, r2+1):
            if pr<=img_size-1:
                ys = np.arange(pr+1)
                startpoint = -pr
            else:
                ys = np.arange(pr-img_size+1, img_size)
                startpoint = pr-2*(img_size-1)
            vals = index_mask[ys, pr-ys]
            if np.sum(vals):
                local_mask[ys[vals],pr-ys[vals],6:8]=get_exts(vals, startpoint, spacing=2)
    return local_mask[:,:,2*nb_start:2*nb_start+2*nb_feature]

def get_exts(vals, startpoint=0, spacing=1):
    exts = np.zeros((len(vals),2), dtype='int32')
    start, end = np.argmax(vals), len(vals) - np.argmax(vals[::-1])
    while not(np.all(vals[start:end])):
        val = np.argmin(vals[start:end])
        assert val!=0
        exts[start:start+val] = [startpoint+spacing*start, 
                                 startpoint+spacing*(start+val-1)]
        start = start+val+np.argmax(vals[start+val:end])
    exts[start:end]=[startpoint+spacing*start,startpoint+spacing*(end-1)]
    assert end!=start
    return exts[vals]

def mask_localpos_old(mask, tp='all'):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    nb_mask = mask.max()
    nb_feature = 4 if tp=='all' else 2
    nb_start = 2 if tp=='di' else 0

    if nb_mask==0:
        return np.zeros(mask.shape+(nb_feature,), dtype='float32')
    else:
        local_mask = extract_local_mask(mask, tp)    

        h, w = mask.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.astype('float32'), y.astype('float32')
        x[mask==0]=0
        y[mask==0]=0
        dr,dl = x+y, y-x
        
        res = [y, x, dr, dl][nb_start:nb_start+nb_feature]        
        index_mask = mask!=0
        mask_x, mask_y = x[index_mask].astype('int32'), y[index_mask].astype('int32')
        for i in range(nb_feature):
            delta = local_mask[mask_y, mask_x,2*i+1]-local_mask[mask_y, mask_x, 2*i]
            res[i][mask_y[delta!=0], mask_x[delta!=0]] = 2*(res[i][mask_y[delta!=0], mask_x[delta!=0]] - 
                                           local_mask[mask_y[delta!=0], mask_x[delta!=0],2*i])/delta[delta!=0]-1
            res[i][mask_y[delta==0], mask_x[delta==0]] = 1e-7
        return np.stack(res, axis=2)
  


def mask_localpos(mask, tp='all', labels0=None, coords=None):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    nb_mask = mask.max()
    nb_feature = 4 if tp=='all' else 2
    nb_start = 2 if tp=='di' else 0
    if nb_mask==0:
        return np.zeros(mask.shape+(nb_feature,), dtype='float32')
    else:
        if labels0 is None:
            labels0 = get_labels(mask.shape[0])
        if coords is None:
            coords = get_coords(mask.shape[0])
        mask0 = np.zeros((mask.shape[0]+2, mask.shape[0]+2), dtype=mask.dtype)
        mask0[1:-1,1:-1] = mask
        
        local_mask = np.zeros(mask.shape+(8,), dtype='float32')
        bbox = extract_bboxes_4di(mask0, tp)   
        for i in range(nb_mask):
            y1, y2, x1, x2 = bbox[i, :4]
            boundary = find_boundaries(mask0[y1-1:y2+2, x1-1:x2+2]==i+1, mode='outer',
                                       connectivity=2)
            boundary_label = labels0[y1-1:y2+2,x1-1:x2+2][boundary]
            mask_label = labels0[y1:y2+1,x1:x2+1][mask0[y1:y2+1,x1:x2+1]==i+1]
            index_mask = mask==i+1
            for nb in range(4):
                boundary_sort = np.sort(boundary_label[:,nb])
                order = np.searchsorted(boundary_sort, mask_label[:,nb], side='right')
                local_mask[index_mask, 2*nb] = coords[(boundary_sort[order-1]+1).astype('int32'), nb]
                local_mask[index_mask, 2*nb+1] = coords[np.ceil(boundary_sort[order]-1).astype('int32'), nb]
                

        h, w = mask.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.astype('float32'), y.astype('float32')
        x[mask==0]=0
        y[mask==0]=0
        dr,dl = x+y, y-x
  
        res = [y, x, dr, dl][nb_start:nb_start+nb_feature]   
        
        index_mask = mask!=0
        mask_x, mask_y = x[index_mask].astype('int32'), y[index_mask].astype('int32')
        for i in range(nb_feature):
            delta = local_mask[mask_y, mask_x,2*i+1]-local_mask[mask_y, mask_x, 2*i]
            if np.sum(delta==0):
                mask_x_nb, mask_y_nb = mask_x[delta!=0], mask_y[delta!=0]
                res[i][mask_y_nb, mask_x_nb] = 2*(res[i][mask_y_nb, mask_x_nb] - 
                                           local_mask[mask_y_nb, mask_x_nb,2*i])/delta[delta!=0]-1
                res[i][mask_y[delta==0], mask_x[delta==0]] = 1e-7
            else:
                res[i][mask_y, mask_x] = 2*(res[i][mask_y, mask_x] - 
                                           local_mask[mask_y, mask_x,2*i])/delta-1
        return np.stack(res, axis=2)