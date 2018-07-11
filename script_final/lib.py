#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:54:48 2018

@author: work
"""
import numpy as np 
import keras.backend as K
from keras.utils import plot_model
from scipy.signal import convolve2d

def delta(pred):
    dpred = np.zeros_like(pred)
    dpred[1:-1,1:-1,0] = pred[2:,2:,0]-pred[:-2,2:,0] + \
                         2*(pred[2:,1:-1,0]- pred[:-2,1:-1,0]) + \
                         pred[2:,:-2,0]-pred[:-2,:-2,0]
    dpred[1:-1,1:-1,1] = pred[2:,2:,1]-pred[2:,:-2,1] + \
                         2*(pred[1:-1,2:,1]-pred[1:-1,:-2,1]) +\
                         pred[:-2,2:,1]-pred[:-2,:-2,1]
    dpred[1:-1,1:-1,2]=  (pred[2:,2:,2]-pred[:-2,:-2,2] +\
                          pred[1:-1,2:,2]-pred[:-2,1:-1,2]+\
                          pred[2:,1:-1,2]-pred[1:-1,:-2,2])*np.sqrt(2)
    dpred[1:-1,1:-1,3]=  (pred[2:,:-2,3]-pred[:-2,2:,3] +\
                          pred[2:,1:-1,3]-pred[1:-1,2:,3]+\
                          pred[1:-1,:-2,3]-pred[:-2,1:-1,3])*np.sqrt(2)
    return dpred

def delta0(pred):
    dpred = np.zeros_like(pred)
    dpred[1:-1,1:-1,0] = 4*(pred[2:,1:-1,0]- pred[:-2,1:-1,0]) 
    dpred[1:-1,1:-1,1] = 4*(pred[1:-1,2:,1]-pred[1:-1,:-2,1])
    dpred[1:-1,1:-1,2]=  2*np.sqrt(2)*(pred[2:,2:,2]-pred[:-2,:-2,2])
    dpred[1:-1,1:-1,3]=  2*np.sqrt(2)*(pred[2:,:-2,3]-pred[:-2,2:,3])
    return dpred
    
def renumber_mask(mask_cut):
    ##renumber mask_cut to values continuous from 0. 
    mask_val = np.unique(mask_cut)  
    mask_val = mask_val[mask_val!=0]
    mask_re = np.zeros_like(mask_cut)
    if len(mask_val):
        inds = np.zeros(mask_val.max()+1, dtype=int)
        inds[mask_val] = np.arange(len(mask_val))+1
        index = mask_cut!=0   
        mask_re[index] = inds[mask_cut[index]]
    return mask_re, mask_val

def plot_png(model, name):
    plot_model(model, to_file='{}.png'.format(name), show_shapes=True)  

def inst_weight_np(output_y, output_x, output_dr, output_dl, config=None):
    dy = output_y[:,2:,2:]-output_y[:, :-2,2:] + \
         2*(output_y[:,2:,1:-1]- output_y[:,:-2,1:-1]) + \
         output_y[:,2:,:-2]-output_y[:,:-2,:-2]
    dx = output_x[:,2:,2:]- output_x[:,2:,:-2] + \
         2*( output_x[:,1:-1,2:]- output_x[:,1:-1,:-2]) +\
         output_x[:,:-2,2:]- output_x[:,:-2,:-2]
    ddr=  (output_dr[:,2:,2:]-output_dr[:,:-2,:-2] +\
           output_dr[:,1:-1,2:]-output_dr[:,:-2,1:-1]+\
           output_dr[:,2:,1:-1]-output_dr[:,1:-1,:-2])*2
    ddl=  (output_dl[:,2:,:-2]-output_dl[:,:-2,2:] +\
           output_dl[:,2:,1:-1]-output_dl[:,1:-1,2:]+\
           output_dl[:,1:-1,:-2]-output_dl[:,:-2,1:-1])*2
    dpred = np.concatenate([dy,dx,ddr,ddl],axis=-1)
    dpred = np.pad(dpred, padding=((0,0),(1,1),(1,1),(0,0)))
    weight_fg = np.all(dpred>config.GRADIENT_THRES, axis=3, keepdims=True)
    
    weight = np.clip(np.sqrt(weight_fg*np.prod(dpred, axis=3, keepdims=True)), 
                    config.WEIGHT_AREA/config.CLIP_AREA_HIGH, 
                    config.WEIGHT_AREA/config.CLIP_AREA_LOW)
    weight +=(1-weight_fg)*config.WEIGHT_AREA/config.BG_AREA
    weight = convolve2d(weight, config.GAUSSIAN_KERNEL[:,:,0,0], 'same')
    return weight

def weighted_mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred[:,:,:,:1])*y_pred[:,:,:,1:2], 
                   axis=(1,2,3))/(np.mean(y_pred[:,:,:,1:2], axis=(1,2,3))+1e-7)

def my_mae2_np(y_true, y_pred):
    weight = (y_true!=0).astype('float32')*y_pred[:,:,:,1:]
    return np.mean(np.abs(y_true - y_pred[:,:,:,:1])*weight, axis=(1,2,3))/(
            np.mean(weight, axis=(1,2,3))+1e-7)
    
def weighted_mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred[:,:,:,:1])*y_pred[:,:,:,1:], 
                  axis=(1,2,3))/(K.mean(y_pred[:,:,:,1:], axis=(1,2,3))+K.epsilon())

    
def my_mae2(y_true, y_pred):
    weight = K.cast(K.not_equal(y_true, 0), K.floatx())
    weight = weight*y_pred[:,:,:,1:]
    return K.mean(K.abs(y_true - y_pred[:,:,:,:1])*weight, axis=(1,2,3))/(K.mean(weight, axis=(1,2,3))+K.epsilon())

def my_brec(y_true, y_pred):
    ntrue = K.cast(K.equal(K.round(y_true), 1), K.floatx())
    npred = K.cast(K.equal(K.round(y_pred), 1), K.floatx())
    return  K.sum(ntrue*npred)/(K.sum(npred)+K.epsilon())

def my_bacc(y_true, y_pred):
    ntrue = K.cast(K.equal(K.round(y_true), 1), K.floatx())
    npred = K.cast(K.equal(K.round(y_pred), 1), K.floatx())
    return  K.sum(ntrue*npred)/(K.sum(ntrue)+K.epsilon())