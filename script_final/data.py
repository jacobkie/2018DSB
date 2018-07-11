#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:22:31 2018

@author: work
"""
import numpy as np
from skimage.segmentation import find_boundaries

def get_isolated(masks):
    res=[]
    for mask in masks:
        nb_mask = mask.max()
        inds = np.zeros((nb_mask+1), dtype='uint8')
        for i in range(nb_mask):
            boundary = find_boundaries(mask==i+1, connectivity=2, mode='outer')
            if np.any(mask[boundary]):
                inds[i+1]=1
        res.append(inds)
    return res

def get_inside(masks):
    res = []
    h,w = np.shape(masks)[1:]
    for mask in masks:
        nb_mask = mask.max()
        inds = np.zeros((nb_mask+1), dtype='uint8')
        for x in [0,h-1]:
            for y in range(w):
                inds[mask[x,y]]=1
        for x in range(h):        
            for y in [0,w-1]:
                inds[mask[x,y]]=1
        res.append(inds)
    return res

def get_mark_properties(images, masks):
    sharps = []
    roughs = []
    intens = []
    areas  = []
    limits = []
    no_boarder=np.zeros_like(masks[0])
    no_boarder[1:-1,1:-1]=1
    w, h = np.shape(masks)[1:]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    for image, mask in zip(images, masks):
        nb_mask = mask.max()
        intensity=np.zeros((nb_mask+1),dtype='float32')
        sharpness=np.zeros((nb_mask+1),dtype='float32')
        roughness=np.zeros((nb_mask+1),dtype='float32')
        area_mask=np.zeros((nb_mask+1),dtype='float32')
        limits_mask = []
        limits_mask.append(np.asarray([0,h-1,0,w-1]))
        gray_scale = np.mean(image, axis = 2)
        intensity[0]=np.mean(gray_scale[mask==0])
        roughness[0]=np.std(gray_scale[mask==0])
        area_mask[0] =np.sum(mask==0)
        for i in range(1,nb_mask+1):
            intensity[i] = np.mean(gray_scale[mask==i])
            roughness[i] = np.std(gray_scale[mask==i])
            boundary = find_boundaries(mask==i, connectivity=2, mode='inner') 
            limits_mask.append(np.asarray([np.min(xs[boundary])-1,np.max(xs[boundary])+2,\
                                           np.min(ys[boundary])-1,np.max(ys[boundary])+2]))            
            boundary = boundary * no_boarder >0
            gradients = []
            for y,x in zip(xs[boundary],ys[boundary]):
                pass
                gx = gray_scale[x+1,y  ] * 2+  gray_scale[x+1,y+1] + gray_scale[x+1,y-1] -\
                     gray_scale[x-1,y  ] * 2-  gray_scale[x-1,y+1] - gray_scale[x-1,y-1]
                gy = gray_scale[x  ,y+1] * 2+  gray_scale[x+1,y+1] + gray_scale[x-1,y+1] -\
                     gray_scale[x  ,y-1] * 2-  gray_scale[x+1,y-1] - gray_scale[x-1,y-1]
                gradients.append(np.sqrt(gx*gx + gy*gy))
            sharpness[i] = np.mean(gradients)                                 
            area_mask[i] = np.sum(mask==i)            
        sharps.append(sharpness)
        roughs.append(roughness)
        intens.append(intensity)
        areas.append(area_mask)
        limits.append(limits_mask)        
    return sharps, roughs, intens, areas, limits   