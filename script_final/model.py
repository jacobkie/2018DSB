# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:32:22 2018

@author: Jackie
"""       
import keras.backend as K                                    
from keras.layers import BatchNormalization, Activation, Input, Dropout, \
                         Conv2D, MaxPooling2D,Conv2DTranspose, Concatenate, \
                         Reshape, Lambda, Maximum
from keras import Model, optimizers,losses, activations, models
from lib import my_bacc, my_brec, weighted_mae, my_mae2
import tensorflow as tf
import numpy as np

def inst_weight(output_y, output_x, output_dr, output_dl, config=None):
    dy = output_y[:,2:,2:]-output_y[:, :-2,2:] + \
         2*(output_y[:,2:,1:-1]- output_y[:,:-2,1:-1]) + \
         output_y[:,2:,:-2]-output_y[:,:-2,:-2]
    dx = output_x[:,2:,2:]- output_x[:,2:,:-2] + \
         2*( output_x[:,1:-1,2:]- output_x[:,1:-1,:-2]) +\
         output_x[:,:-2,2:]- output_x[:,:-2,:-2]
    ddr=  (output_dr[:,2:,2:]-output_dr[:,:-2,:-2] +\
           output_dr[:,1:-1,2:]-output_dr[:,:-2,1:-1]+\
           output_dr[:,2:,1:-1]-output_dr[:,1:-1,:-2])*K.constant(2)
    ddl=  (output_dl[:,2:,:-2]-output_dl[:,:-2,2:] +\
           output_dl[:,2:,1:-1]-output_dl[:,1:-1,2:]+\
           output_dl[:,1:-1,:-2]-output_dl[:,:-2,1:-1])*K.constant(2)
    dpred = K.concatenate([dy,dx,ddr,ddl],axis=-1)
    dpred = K.spatial_2d_padding(dpred)
    weight_fg = K.cast(K.all(dpred>K.constant(config.GRADIENT_THRES), axis=3, 
                          keepdims=True), K.floatx())
    
    weight = K.clip(K.sqrt(weight_fg*K.prod(dpred, axis=3, keepdims=True)), 
                    config.WEIGHT_AREA/config.CLIP_AREA_HIGH, 
                    config.WEIGHT_AREA/config.CLIP_AREA_LOW)
    weight +=(1-weight_fg)*config.WEIGHT_AREA/config.BG_AREA
    weight = K.conv2d(weight, K.constant(config.GAUSSIAN_KERNEL),
                      padding='same')
    return K.stop_gradient(weight)

