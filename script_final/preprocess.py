#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This code is copied from https://github.com/vsvinayak/mnist-helper.
It requires Scipy to perform convolve2d.
Default parameters are modified.
"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from skimage.segmentation import find_boundaries
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
import scipy
from skimage.transform import AffineTransform
from keras.preprocessing.image import transform_matrix_offset_center
from cv2 import GaussianBlur
from data import get_mark_properties,get_inside,get_isolated
from lib import renumber_mask
import cv2

def insert(image_batch, mask_batch, max_tries, max_fails):
    sharps, roughs, intens, areas, limits =  get_mark_properties(image_batch, mask_batch)
    inside = get_inside(mask_batch)     # True if not inside
    isolated = get_isolated(mask_batch) # True if not isolated
    new_image_batch = image_batch.astype('int16')
    new_mask_batch  = mask_batch.astype('uint8')
    w, h = np.shape(mask_batch)[1:]
    for nb in range(image_batch.shape[0]):   
        image, mask = new_image_batch[nb], new_mask_batch[nb]
        nb_mask = mask.max()
        if nb_mask==0:
            continue
        negative = - np.mean(intens[nb][1:]) + intens[nb][0]
        negative = negative > 0

        new_nb_mask = nb_mask
        tries, fails = 0, 0
        area_bg = areas[nb][0]
        #available_masks = (inside[nb] + isolated[nb])==0
        while (tries < max_tries and fails < max_fails and area_bg/w/h>0.7):
            tries+=1
            id = np.random.randint(1,nb_mask+1)
            if inside[nb][id] or isolated[nb][id] or ((np.abs(intens[nb][id]-intens[nb][0])<
                     np.minimum(intens[nb][id],intens[nb][0])) and not negative) or (negative and 
                     np.abs(intens[nb][id]-intens[nb][0]) <15 ) or sharps[nb][id]<5:
                continue
            bbox = limits[nb][id]
            bw, bh = bbox[1]-bbox[0], bbox[3]-bbox[2]
            image_pad  = image_batch[nb,bbox[2]:bbox[3],bbox[0]:bbox[1],:]
            x = np.random.randint(0,w-bw)
            y = np.random.randint(0,h-bh)
            mask_pad = (mask_batch[nb, bbox[2]:bbox[3],bbox[0]:bbox[1]] ==id)                
            mask_pad_old = mask[y:y+bh, x:x+bw]
            labels_old =np.unique(mask_pad_old[mask_pad])
            flag = 0
            for label in labels_old:
                if label ==0:
                    continue
                overlap=np.sum((mask_pad_old==label)*(mask_pad))
                oldlabelarea=np.sum(mask==label)
                if  overlap/oldlabelarea > np.arctan(oldlabelarea/100)/3:
                    flag = 1                    
                    break
            if flag==1:
                fails+=1
                break
            if np.random.rand(1)>0.5:
                image_pad = image_pad[:,::-1]
                mask_pad = mask_pad[:,::-1]
            if np.random.rand(1)>0.5:
                image_pad = image_pad[::-1]
                mask_pad = mask_pad[::-1]                
            boundary1 = find_boundaries(mask_pad, connectivity=2, mode='outer')
            boundary2 = find_boundaries(mask_pad, connectivity=2, mode='inner')
            weight=mask_pad.astype('float32')
            weight[boundary1]=1/3
            weight[boundary2]=2/3
            weight=GaussianBlur(weight, (3, 3), sigmaX=10)
            new_nb_mask +=1 
            trans = np.random.rand(1)*0.4+0.6 
            image[y:y+bh,x:x+bw,:] = image_pad*np.expand_dims(weight,axis=2) * trans +\
                  image[y:y+bh,x:x+bw,:] * (1-np.expand_dims(weight,axis=2) * trans)
            area_bg -= np.sum(mask_pad>0)
            mask[y:y+bh,x:x+bw][mask_pad] = new_nb_mask    
            if np.max(image)>255:
                new_image_batch[nb]= 1.0*image/np.max(image)*255
            
        if mask.max()+1 != len(np.unique(mask)):
            mask, _ = renumber_mask(mask)
            assert mask.max()+1 == len(np.unique(mask))
    return new_image_batch, new_mask_batch

def enhance(image_batch, mask_batch, max_tries = 10, max_enhance_ratio=.8):
    new_image_batch = image_batch.astype('int16')
    new_mask_batch  = mask_batch.astype('uint8')
    for nb in range(image_batch.shape[0]):   
        image, mask = new_image_batch[nb], new_mask_batch[nb]
        nb_mask = mask.max()
        enhanced = np.zeros(nb_mask+1)
        if nb_mask==0:
            continue
        negative =  np.mean(image[mask>0]) - np.mean(image[mask == 0])  <0
        tries = 0
        if negative:
            while (tries < max_tries and tries < nb_mask/2):
                tries +=1
                id = np.random.randint(1,nb_mask+1)
                if enhanced[id]>0:
                    continue
                enhanced[id]=1
                id_mask = (mask ==id) 
                if np.sum(id_mask)<40:
                    continue
                boundary1 = find_boundaries(id_mask, connectivity=2, mode='outer')
                boundary2 = find_boundaries(id_mask, connectivity=2, mode='inner')
                boundary3 = find_boundaries((id_mask*1.0 - boundary2) >0, connectivity=2, mode='inner')
                weight=id_mask.astype('float32')
                weight[boundary1]=1/4
                weight[boundary2]=1/2
                weight[boundary3]=3/4
                enhance_ratio = (np.random.rand(1)-.1) * np.minimum(max_enhance_ratio, 254.99/(256-np.max(image[id_mask]))-1.1)
                weight = weight * enhance_ratio + 1
                weight=GaussianBlur(weight, (3, 3), sigmaX=10)
                image = 255 -(255 - image) * np.expand_dims(weight,axis=2)
        image[image>255] = 255
        image[image<0] = 0
        new_image_batch[nb] = image
    return new_image_batch   

def stretch(image_batch, mask_batch,max_ratio = 2):
    new_image_batch = image_batch.astype('int16')
    new_mask_batch  = mask_batch.astype('uint8')    
    for nb in range(image_batch.shape[0]):
        while True:
            x1,y1,x2,y2= np.random.randint(0,256,size=(4))
            if np.abs(x2-x1)>=128 and np.abs(y2-y1)>=128:
                break
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        new_image_batch[nb]=cv2.resize(image_batch[nb,y1:y2+1,x1:x2+1], dsize =(256, 256))
        mask=cv2.resize(mask_batch[nb,y1:y2+1,x1:x2+1], dsize =(256, 256), 
                      interpolation=cv2.INTER_NEAREST)
        new_mask_batch[nb],_ = renumber_mask(mask)
    return new_image_batch, new_mask_batch            

def level_noise(image_batch, mask_batch, max_level_ratio=4, max_noise_ratio=.05):
    new_image_batch = image_batch.astype('int16')
    for nb in range(image_batch.shape[0]):   
        image, mask = new_image_batch[nb], mask_batch[nb]
        nb_mask = mask.max()
        if nb_mask==0:
            continue        
        positive =  np.mean(image[mask>0]) - np.mean(image[mask == 0])  >0
        
        if positive:
            image = (image+1)*(np.minimum(0.8 + (np.random.rand(1))*\
                    (255/np.max(image)-.8), max_level_ratio)) *\
                    np.expand_dims(np.random.rand(256,256)*max_noise_ratio +1-max_noise_ratio/2,axis=2) +\
                    np.random.randint(20)-10
        else:
            image = 256.0- (255.0-image)*np.minimum(0.5 + np.random.rand(1)*\
                      (280.0/(255.0-np.min(np.mean(image,axis=2)))-.5),max_level_ratio) *\
                    np.expand_dims(np.random.rand(256,256)*max_noise_ratio +1-max_noise_ratio/2,axis=2) +\
                    np.random.randint(20)-10            
        image[image>255] = 255
        image[image<0] = 0
        new_image_batch[nb] = image
    return new_image_batch    

    
def get_affine_transform_matrix(h, w, rotation_range=0, width_shift_range=0,
                                height_shift_range=0, shear_range=0, zoom_range=0):
    rotation = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    wshift = np.random.uniform(-width_shift_range, width_shift_range)*w
    hshift = np.random.uniform(-height_shift_range,height_shift_range)*h
    shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
    if np.isscalar(zoom_range):
        zoom_range = [1, zoom_range]
    hzoom, wzoom = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    
    matrix = AffineTransform(scale = (wzoom, hzoom), rotation = rotation,
                             shear = shear, translation = (wshift, hshift))._inv_matrix
    matrix = transform_matrix_offset_center(matrix, h, w)
    return matrix

def affine_transform_batch(image_batch, mask_batch, rotation_range=0, width_shift_range=0,
                           height_shift_range=0, shear_range=0, zoom_range=0):

    _, h, w, _ = image_batch.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.astype('float32'), y.astype('float32')

    matrix = get_affine_transform_matrix(h, w, rotation_range = rotation_range,
                                         width_shift_range=width_shift_range,
                                         height_shift_range=height_shift_range, 
                                         shear_range=shear_range, zoom_range=zoom_range)
    xt = scipy.ndimage.affine_transform(x, matrix, order=1, cval=-512)
    yt = scipy.ndimage.affine_transform(y, matrix, order=1, cval=-512)

    image_affine = image_transform_batch(image_batch, (yt, xt))
    mask_affine = mask_transform_batch_fast(mask_batch, (yt, xt))
    return image_affine, mask_affine

def image_transform_batch(image_batch, coordinates):
    image_batch_trans = image_batch.copy()
    for nb_batch in range(image_batch.shape[0]):
        for z in range(image_batch.shape[3]):
            image_batch_trans[nb_batch, :,:,z] = map_coordinates(image_batch[nb_batch,:,:,z], 
                             coordinates, order=1, mode='constant')
    return image_batch_trans

def mask_transform_batch_fast(mask_batch, coordinates):
    mask_batch_trans = mask_batch.copy()
    for nb_batch in range(mask_batch.shape[0]):
        mask_batch_trans[nb_batch] = map_coordinates(mask_batch[nb_batch], coordinates, 
                        order=0, mode='constant')
    return mask_batch_trans

def affine_transform(image, mask, rotation_range=0, width_shift_range=0,
                     height_shift_range=0, shear_range=0, zoom_range=0):

    h, w, _ = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    matrix = get_affine_transform_matrix(h, w, rotation_range = rotation_range,
                                         width_shift_range=width_shift_range,
                                         height_shift_range=height_shift_range, 
                                         shear_range=shear_range, zoom_range=zoom_range)
    xt = scipy.ndimage.affine_transform(x, matrix, order=1, cval=-512)
    yt = scipy.ndimage.affine_transform(y, matrix, order=1, cval=-512)

    image_affine = image_transform(image, (yt, xt))
    mask_affine = mask_transform_fast(mask, (yt, xt))
    return image_affine, mask_affine

def image_transform(image, coordinates):
    image_trans = image.copy()
    for z in range(image.shape[2]):
        image_trans[:,:,z] = map_coordinates(image[:,:,z], coordinates, 
           order=1, mode='constant')
    return image_trans

def mask_transform_fast(mask, coordinates):
    mask_trans = map_coordinates(mask, coordinates, order=0, mode='constant')
    return mask_trans

def mask_transform(mask, coordinates):
    h, w = mask.shape
    nb_mask = mask.max()

    yt, xt = coordinates
    y_floor, x_floor = np.floor(yt), np.floor(xt)    

    score = np.zeros((h, w, nb_mask,))    
    for (y_shift, x_shift) in [(0,0),(0,1),(1,0),(1,1)]:
        mask_index = map_coordinates(mask, (y_floor+y_shift, x_floor+x_shift), 
             order=0, mode='constant')
        index = mask_index!=0
        dist = np.sqrt((yt[index]-y_floor[index]-y_shift)**2+(xt[index]-x_floor[index]-x_shift)**2)
        score[index, mask_index[index]-1] += 1/dist
 
    index = score.sum(2)!=0
    mask_trans = np.zeros_like(mask)
    mask_trans[index]= score[index].argmax(1)+1
    return mask_trans

def mask_transform_slow(mask, coordinates):
    h, w = mask.shape
    nb_mask = mask.max()
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    yt, xt = coordinates
    y_floor, x_floor = np.floor(yt), np.floor(xt)    
    
    score = np.zeros((h, w, nb_mask+1,))    
    for (y_shift, x_shift) in [(0,0),(0,1),(1,0),(1,1)]:
        mask_index = map_coordinates(mask, (y_floor+y_shift, x_floor+x_shift), 
             order=0, mode='constant')
        dist = np.sqrt((yt-y_floor-y_shift)**2+(xt-x_floor-x_shift)**2)
        score[y,x, mask_index] += 1/dist
 
    mask_trans= score.argmax(2)
    return mask_trans

def elastic_transform(image, mask, alpha, sigma):     
    h, w = mask.shape
                 
    dx = gaussian_filter((np.random.rand(h,w) * 2 - 1), sigma, mode='constant') * alpha
    dy = gaussian_filter((np.random.rand(h,w) * 2 - 1), sigma, mode='constant') * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    yt, xt = y+dy, x+dx

    image_trans = image_transform(image, (yt, xt))
    mask_trans = mask_transform_fast(mask, (yt, xt))
    return image_trans, mask_trans

def elastic_transform_batch(image_batch, mask_batch, alpha, sigma):     
    _, h, w = mask_batch.shape
    alpha_ = np.random.uniform(0, alpha)
    
    dx = gaussian_filter((np.random.rand(h,w) * 2 - 1), sigma, mode='constant') * alpha_
    dy = gaussian_filter((np.random.rand(h,w) * 2 - 1), sigma, mode='constant') * alpha_
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    yt, xt = y+dy, x+dx

    image_trans = image_transform_batch(image_batch, (yt, xt))
    mask_trans = mask_transform_batch_fast(mask_batch, (yt, xt))
    return image_trans, mask_trans

def draw_grid(im, grid_size):
    # Draw grid lines
    color = im.max()
    for i in range(0, im.shape[1], grid_size):
        im[:,i,:] = color
    for j in range(0, im.shape[0], grid_size):
        im[j,:,:] = color  
