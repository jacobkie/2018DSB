# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 09:52:46 2018

@author: Jackie
"""
import numpy as np

from skimage.morphology import label, binary_dilation,binary_erosion,remove_small_holes
from scipy.ndimage import generate_binary_structure

from load import mask_12m_no, mask_12m
from lib import delta

def postprocess(preds, config):
    assert preds.shape[2]==5
    ldelta = delta(preds[:,:,1:])
    #ldelta = delta0(preds[:,:,5:])
    connected = np.all(ldelta>config.GRADIENT_THRES, 2)
    base = connected * (preds[:,:,0]>config.MASK_THRES)
    
    wall = np.sum(np.abs(preds[:,:,1:]),axis = -1)   
    base_label = label(base) 
    vals, counts = np.unique(base_label[base_label>0], return_counts=True)
    for val in vals[(counts<config.CLIP_AREA_LOW)]:
        base_label[base_label==val]=0
    vals = vals[(counts>=config.CLIP_AREA_LOW)]

    for val in vals:
        
        label_mask = base_label == val   
        if np.sum(label_mask)==0:
            continue
        label_mask = remove_small_holes(label_mask)
        label_mask = basin(label_mask, wall)
        label_mask = remove_small_holes(label_mask)

        '''
        label_bdr = label_mask^binary_erosion(label_mask)
        min_wall = np.min(wall[label_mask])
        ave_bdr_wall = np.mean(wall[label_bdr])
        if ave_bdr_wall < min_wall + config.WALL_DEPTH:
            label_mask = 0
        '''
        base_label[label_mask] = val

    vals, counts = np.unique(base_label[base_label>0], return_counts=True)
    for val in vals[(counts<config.CLIP_AREA_LOW)]:
        base_label[base_label==val]=0        
    return base_label

def modify_w_unet(base_label, preds, thres=0.25):
    base_label = base_label.copy()
    vals = np.unique(base_label[base_label>0])
    struct = generate_binary_structure(2,2)
    for nb_dilation in range(3):    
        for val in vals:
            label_mask = base_label==val
            base_label[binary_dilation(label_mask,struct)&(preds[:,:,0]>thres)&(base_label==0)]=val
    return base_label
 
def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    if np.sum(union)>0:
        overlaps = intersections / union
    else:
        return 0
    return overlaps

def compute_mean_iou(overlaps, return_list=False):
    miou = []
    overlaps *= (overlaps>=.5)
    for step in np.arange(0.5,1,0.05):
        overlap_step = overlaps >=step
        tp = np.sum(np.sum(overlap_step,axis=0)>0)
        fp = np.sum(np.sum(overlap_step,axis=0)==0)
        fn = np.sum(np.sum(overlap_step,axis=-1)==0)
        miou.append(tp/(tp+fp+fn))
    if return_list:
        return miou
    else:
        return np.mean(miou)
    
def get_score(base_label, mask, return_list = False):
    if (mask.max()==0)|(base_label.max()==0):
        if (mask.max()==0)&(base_label.max()==0):
            return 1
        else:
            return 0
    mask_pred = mask_12m_no(base_label)
    mask_true = mask_12m(mask)    
    overlaps = compute_overlaps_masks(mask_true, mask_pred)
    return compute_mean_iou(overlaps, return_list = return_list)

def map_coords(array, coords):
    h, w = array.shape
    y, x = coords
    res = np.zeros(y.shape, dtype=array.dtype)
    outb = np.any([y<0, y>h-1, x<0, x>w-1], axis=0)
    y, x = np.clip(y, 0, h-1), np.clip(x, 0, w-1)
    
    res = array[y, x]
    res[outb]=0
    return res

def basin(label_mask, wall):
    h,w = np.shape(label_mask)
    y, x = np.mgrid[0:h, 0:w]
    struct = generate_binary_structure(2,2)
    shifty, shiftx = np.mgrid[0:3, 0:3]
    shifty = (shifty-1).flatten()
    shiftx = (shiftx-1).flatten()

    for i in range(4):
        obdr = label_mask^binary_dilation(label_mask, struct)
        ibdr = label_mask^binary_erosion(label_mask, struct)
        yob, xob = y[obdr], x[obdr]        
        ynb, xnb = yob.reshape(-1,1)+shifty, xob.reshape(-1,1)+shiftx
        
        wallnb = np.min(map_coords(wall, (ynb, xnb))*(map_coords(ibdr, (ynb, xnb))==1)+\
                        5*(map_coords(ibdr, (ynb, xnb))!=1),1)
        keep = (wall[yob,xob]>wallnb)&(wallnb<=4)
        label_mask[yob[keep], xob[keep]]=True
        if np.sum(keep)==0:
            break
    return label_mask


    

