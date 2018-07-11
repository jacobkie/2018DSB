#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:10:27 2018

@author: work
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries

import cv2
from model_rcnn_weight import UnetRCNN as model_rcnn
from params import DsbConfig
from load import load_from_cache, save_to_cache, get_ax, save_file, load_file, \
                mask_12m_no, mask_12m, save_to_cache_multi, k2, load_from_cache_multi
from postprocess0 import postprocess, modify_w_unet, get_score
from lib import renumber_mask
from submission import trans2square
from mask_pos import mask_localpos

def get_inputs_rcnn(image):
    batch_size, img_size = image.shape[:2]    
    mask_weight = np.ones(image.shape[:3] + (1,), dtype='float32')
    inp = np.zeros((batch_size, img_size*4, img_size*4, 3), dtype='float32')
    for nb in range(batch_size):
        inp[nb] = cv2.resize(image[nb], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return [inp, mask_weight]


                
def predict_test_rcnn(df_name, weight_dir, weight_fl = None):
    config = DsbConfig()
    config.BATCH_SIZE = 1
    config.GPU_COUNT = 1

    max_shape = 1024
    unet_ratio, tp = 1, 'all'
    df = load_from_cache(df_name)   

    output_names =  ['mask', 'ly','lx','ldr','ldl']
    nb_outputs = [0, 9, 10, 11, 12]
    preds_df = pd.DataFrame(index = df.index, columns = output_names)
    vals = np.unique(df['shape'])

    if weight_fl is None:
        weight_fls = glob.glob(os.path.join(weight_dir, '*.hdf5'))
        weight_fls = sorted(weight_fls, key = lambda x: float(os.path.basename(x)[:-5].split('_')[-1]))
        weight_fl = weight_fls[0]

    for nb, shape in enumerate(vals):
        ind_shape = df[df['shape']==shape].index
        if max(shape)<=max_shape:
            new_shape = 64*int(np.ceil(max(shape)/64)) 
        else:
            new_shape = 512*int(np.ceil(max(shape)/512))             
        print('{}/{}'.format(nb, len(vals)), len(ind_shape), shape, new_shape)
        model = model_rcnn(tp, unet_ratio, 'sgd', config, weight_dir, 
                           min(new_shape, max_shape))
        #model = model_unet(new_shape, unet_ratio, tp, config=config)   
        model.load_weights(weight_fl)            
        model.compile(1e-3)
        images = np.stack(df.loc[ind_shape, 'image'], 0)
        
        if (new_shape, new_shape)!=shape:
            y1, x1 = (new_shape - images.shape[1])//2, (new_shape - images.shape[2])//2
            y2, x2 = new_shape - images.shape[1]-y1, new_shape - images.shape[2]-x1
            images = np.pad(images, ((0,0),(y1,y2),(x1,x2),(0,0)), mode='constant', 
                            constant_values = 0)
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0
        X = get_inputs_rcnn(images)

        if new_shape>max_shape:
            nb_cut = int(np.ceil(new_shape/512))-1
            y_preds = [np.zeros((images.shape[0], new_shape, new_shape, 1), dtype='float32')]+\
                      [np.zeros((images.shape[0], new_shape, new_shape, 2), dtype='float32') for _ in range(12)]
            for nb_y in range(nb_cut):
                start_y, end_y = 512*nb_y, 512*(nb_y+2)
                shift_start_y = 0 if nb_y==0 else 256
                shift_end_y = 0 if nb_y==nb_cut-1 else -256
                for nb_x in range(nb_cut):
                    start_x, end_x = 512*nb_x, 512*(nb_x+2)
                    shift_start_x = 0 if nb_x==0 else 256
                    shift_end_x = 0 if nb_x==nb_cut-1 else -256

                    print(start_y, end_y, start_x, end_x)
                    print(shift_start_y, shift_end_y, shift_start_x, shift_end_x)
                    X_nb = [X[0][:,4*start_y:4*end_y, 4*start_x:4*end_x],
                            X[1][:,start_y:end_y, start_x:end_x]]
                    preds = model.keras_model.predict(X_nb, batch_size = 1, verbose=1)
                    for nb_output in range(13):
                        y_preds[nb_output][:, start_y+shift_start_y:end_y+shift_end_y, 
                               start_x+shift_start_x:end_x+shift_end_x]=\
                               preds[nb_output][:,shift_start_y:max_shape+shift_end_y,
                                    shift_start_x:max_shape+shift_end_x]
        else:
            y_preds = model.keras_model.predict(X, batch_size = 1, verbose=1)

        for nb_output, output_name in zip(nb_outputs, output_names): 
            y_pred = y_preds[nb_output][:,:,:,:1]
            if (new_shape, new_shape)!=shape:
                y_pred = y_pred[:,y1:new_shape-y2, x1:new_shape-x2, :1]
            preds_df.loc[ind_shape, output_name] = list(y_pred.astype('float16'))

    if len(preds_df)>500:
        save_to_cache_multi(preds_df, os.path.join(weight_dir, df_name), len(preds_df)//500+1)
    else:
        save_to_cache(preds_df, os.path.join(weight_dir, df_name))

def predict_test_rcnn_half(df_name, weight_dir, weight_fl = None):
    config = DsbConfig()
    config.BATCH_SIZE = 1
    config.GPU_COUNT = 1
    unet_ratio, tp = 1, 'all'
    df = load_from_cache(df_name)   
    df['shape'] = df['image'].apply(lambda x: (x.shape[0]//2, x.shape[1]//2, x.shape[2]))
  
    output_names =  ['mask','ly','lx','ldr','ldl']
    nb_outputs = [0, 9, 10, 11, 12]    
    preds_df = pd.DataFrame(index = df.index, columns = output_names)
    vals = np.unique(df['shape'])

    if weight_fl is None:
        weight_fls = glob.glob(os.path.join(weight_dir, '*.hdf5'))
        weight_fls = sorted(weight_fls, key = lambda x: float(os.path.basename(x)[:-5].split('_')[-1]))
        weight_fl = weight_fls[0]

    for nb, shape in enumerate(vals):
        ind_shape = df[df['shape']==shape].index

        new_shape = 64*int(np.ceil(max(shape)/64))
        print('{}/{}'.format(nb, len(vals)), len(ind_shape), shape, new_shape)
        model = model_rcnn(tp, unet_ratio, 'sgd', config, weight_dir, new_shape)
        #model = model_unet(new_shape, unet_ratio, tp, config=config)   
        model.load_weights(weight_fl)            
        model.compile(1e-3)

        images = np.stack([cv2.resize(image, dsize=(shape[1],shape[0]), 
                                      interpolation=cv2.INTER_LINEAR) 
                            for image in df.loc[ind_shape, 'image']], 0)            
        if (new_shape, new_shape)!=shape[:2]:
            y1, x1 = (new_shape - images.shape[1])//2, (new_shape - images.shape[2])//2
            y2, x2 = new_shape - images.shape[1]-y1, new_shape - images.shape[2]-x1
            images = np.pad(images, ((0,0),(y1,y2),(x1,x2),(0,0)), mode='constant', 
                            constant_values = 0)
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0
        inputs = get_inputs_rcnn(images)
        y_preds = model.keras_model.predict(inputs, batch_size = 1, verbose=1)
    
        for nb_output, output_name in zip(nb_outputs, output_names): 
            y_pred = y_preds[nb_output][:,:,:,:1]
            if (new_shape, new_shape)!=shape:
                y_pred = y_pred[:,y1:new_shape-y2, x1:new_shape-x2, :1]
            preds_df.loc[ind_shape, output_name] = list(y_pred.astype('float16'))

    save_to_cache(preds_df, os.path.join(weight_dir, '{}_half'.format(df_name)))

def predict_test_rcnn_quarter(df_name, weight_dir, weight_fl=None):
    config = DsbConfig()
    config.BATCH_SIZE = 1
    config.GPU_COUNT = 1
    unet_ratio, tp = 1, 'all'
    df = load_from_cache(df_name)   
    df['shape'] = df['image'].apply(lambda x: (x.shape[0]//4, x.shape[1]//4, x.shape[2]))

    output_names =  ['mask','ly','lx','ldr','ldl']
    nb_outputs = [0, 9, 10, 11, 12]    
    preds_df = pd.DataFrame(index = df.index, columns = output_names)
    vals = np.unique(df['shape'])

    if weight_fl is None:
        weight_fls = glob.glob(os.path.join(weight_dir, '*.hdf5'))
        weight_fls = sorted(weight_fls, key = lambda x: float(os.path.basename(x)[:-5].split('_')[-1]))
        weight_fl = weight_fls[0]

    for nb, shape in enumerate(vals):
        ind_shape = df[df['shape']==shape].index

        new_shape = 64*int(np.ceil(max(shape)/64))
        print('{}/{}'.format(nb, len(vals)), len(ind_shape), shape, new_shape)
        model = model_rcnn(tp, unet_ratio, 'sgd', config, weight_dir, new_shape)
        #model = model_unet(new_shape, unet_ratio, tp, config=config)   
        model.load_weights(weight_fl)            
        model.compile(1e-3)

        images = np.stack([cv2.resize(image, dsize=(shape[1],shape[0]), 
                                      interpolation=cv2.INTER_LINEAR) 
                            for image in df.loc[ind_shape, 'image']], 0)            
        if (new_shape, new_shape)!=shape[:2]:
            y1, x1 = (new_shape - images.shape[1])//2, (new_shape - images.shape[2])//2
            y2, x2 = new_shape - images.shape[1]-y1, new_shape - images.shape[2]-x1
            images = np.pad(images, ((0,0),(y1,y2),(x1,x2),(0,0)), mode='constant', 
                            constant_values = 0)
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0
        inputs = get_inputs_rcnn(images)
        y_preds = model.keras_model.predict(inputs, batch_size = 1, verbose=1)
        for nb_output, output_name in zip(nb_outputs, output_names): 
            y_pred = y_preds[nb_output][:,:,:,:1]
            if (new_shape, new_shape)!=shape:
                y_pred = y_pred[:,y1:new_shape-y2, x1:new_shape-x2, :1]
            preds_df.loc[ind_shape, output_name] = list(y_pred.astype('float16'))

    save_to_cache(preds_df, os.path.join(weight_dir, '{}_quarter'.format(df_name)))
    
def predict_test_rcnn_two(df_name, weight_dir, weight_fl = None, start=0, end =100,
                          start_run=0):
    config = DsbConfig()
    config.BATCH_SIZE = 1
    config.GPU_COUNT = 1
    max_shape = 1024
    unet_ratio, tp = 1, 'all'
    df = load_from_cache(df_name)   
    df['shape'] = df['image'].apply(lambda x: (x.shape[0]*2, x.shape[1]*2, x.shape[2]))

    output_names =  ['mask', 'ly','lx','ldr','ldl']
    nb_outputs = [0, 9, 10, 11, 12]        
    vals = np.unique(df['shape'])

    if weight_fl is None:
        weight_fls = glob.glob(os.path.join(weight_dir, '*.hdf5'))
        weight_fls = sorted(weight_fls, key = lambda x: float(os.path.basename(x)[:-5].split('_')[-1]))
        weight_fl = weight_fls[0]    
        
    for nb, shape in enumerate(vals):
        if (nb<start)|(nb>=end):
            continue
        if max(shape)<=max_shape:
            new_shape = 64*int(np.ceil(max(shape)/64)) 
        else:
            new_shape = 512*int(np.ceil(max(shape)/512))        
        ind_shape = df[df['shape']==shape].index
        print('{}/{}'.format(nb, len(vals)), len(ind_shape), shape, new_shape)

        if len(ind_shape)*(new_shape//512)**2>800:
            nb_run = (len(ind_shape)*(new_shape//512)**2)//800+1
            size = int(len(ind_shape)/nb_run)+1
            ind_shape0 = ind_shape.copy()
        else:
            nb_run = 1
            size = len(ind_shape)
            
        for run in range(nb_run):
            if run<start_run:
                continue
            if nb_run!=1:
                start, end = run*size, min((run+1)*size, len(ind_shape0))
                ind_shape = ind_shape0[start:end]
            
            preds_df = pd.DataFrame(index = df.index[ind_shape], columns = output_names)            
    
            model = model_rcnn(tp, unet_ratio, 'sgd', config, weight_dir, min(new_shape, max_shape))
            #model = model_unet(new_shape, unet_ratio, tp, config=config)   
            model.load_weights(weight_fl)            
            model.compile(1e-3)
    
            images = np.stack([cv2.resize(image, dsize=(shape[1],shape[0]), 
                                          interpolation=cv2.INTER_LINEAR) 
                                for image in df.loc[ind_shape, 'image']], 0)  
            print(images.shape)
            if (new_shape, new_shape)!=shape[:2]:
                y1, x1 = (new_shape - images.shape[1])//2, (new_shape - images.shape[2])//2
                y2, x2 = new_shape - images.shape[1]-y1, new_shape - images.shape[2]-x1
                images = np.pad(images, ((0,0),(y1,y2),(x1,x2),(0,0)), mode='constant', 
                                constant_values = 0)
            else:
                y1, x1, y2, x2 = 0, 0, 0, 0
            X = get_inputs_rcnn(images)
    
            if new_shape>max_shape:
                nb_cut = int(np.ceil(new_shape/512))-1
                y_preds = [np.zeros((images.shape[0], new_shape, new_shape, 1), dtype='float32')]+\
                          [np.zeros((images.shape[0], new_shape, new_shape, 2), dtype='float32') for _ in range(4)]
                for nb_y in range(nb_cut):
                    start_y, end_y = 512*nb_y, 512*(nb_y+2)
                    shift_start_y = 0 if nb_y==0 else 256
                    shift_end_y = 0 if nb_y==nb_cut-1 else -256
                    for nb_x in range(nb_cut):
                        start_x, end_x = 512*nb_x, 512*(nb_x+2)
                        shift_start_x = 0 if nb_x==0 else 256
                        shift_end_x = 0 if nb_x==nb_cut-1 else -256
    
                        print(start_y, end_y, start_x, end_x)
                        print(shift_start_y, shift_end_y, shift_start_x, shift_end_x)
                        X_nb = [X[0][:,4*start_y:4*end_y, 4*start_x:4*end_x],
                                X[1][:,start_y:end_y, start_x:end_x]]
                        preds = model.keras_model.predict(X_nb, batch_size = 1, verbose=1)
                        for i, nb_output in enumerate(nb_outputs):
                            y_preds[i][:, start_y+shift_start_y:end_y+shift_end_y, 
                                   start_x+shift_start_x:end_x+shift_end_x]=\
                                   preds[nb_output][:,shift_start_y:max_shape+shift_end_y,
                                        shift_start_x:max_shape+shift_end_x]
                        del preds
            else:
                y_preds = model.keras_model.predict(X, batch_size = 1, verbose=1)
                y_preds = [y_preds[i] for i in nb_outputs]
            
            for i, output_name in enumerate(output_names): 
                y_pred = y_preds[i][:,:,:,:1]
                if (new_shape, new_shape)!=shape:
                    y_pred = y_pred[:,y1:new_shape-y2, x1:new_shape-x2, :1]
                preds_df.loc[ind_shape, output_name] = list(y_pred.astype('float16'))
                
            if nb_run==1:
                save_to_cache(preds_df, os.path.join(weight_dir, '{}_two_{}'.format(df_name,nb)))
            else:
                save_to_cache(preds_df, os.path.join(weight_dir, '{}_two_{}_{}'.format(df_name,
                                                           nb, run)))
                
    if len(df)<200:
        preds_df = load_from_cache_multi(os.path.join(weight_dir, '{}_two'.format(df_name)),
                                         len(vals))
        save_to_cache(preds_df, os.path.join(weight_dir, '{}_two'.format(df_name)))
        for nb in range(len(vals)):
            os.remove(os.path.join(weight_dir, '{}_two_{}.dat'.format(df_name, nb)))
            

def get_cons_local_single(mask, preds):    
    mask_s, y1, y2, x1, x2 = trans2square(mask)
    local_true = mask_localpos(mask_s)[y1:mask_s.shape[0]-y2, x1:mask_s.shape[1]-x2]
    local_pred = preds[:,:,5:]
    #local_pred = np.concatenate(preds[['y_true','x_true','dr_true','dl_true']],2)
    #local_pred = np.concatenate(preds[['y','x','dr','dl']],2)
    
    index = mask!=0
    score = np.mean((local_pred[index]-local_true[index])**2)
        
    return score


       
    
def get_mask(df_name, config, weight_dir, save_name, n_jobs = 18):
    save_dir = os.path.join(weight_dir, save_name)
    os.path.exists(save_dir)
    #df = load_from_cache(df_name)   
    
    fl_name = os.path.join(weight_dir, df_name)
    if os.path.exists('{}.dat'.format(fl_name)):
        preds_df = load_from_cache(fl_name)
    else:
        fl_names = glob.glob(os.path.join(weight_dir, '{}_[0-9+].dat'.format(df_name)))
        preds_df = load_from_cache_multi(fl_name, nb=len(fl_names))
    print(preds_df.shape)
    masks0 = Parallel(n_jobs)(delayed(postprocess)(np.concatenate(preds_df.loc[ind, 
                      ['mask','ly','lx','ldr','ldl']], -1), config) for ind in preds_df.index)
    masks0 = Parallel(n_jobs)(delayed(renumber_mask)(mask) for mask in masks0)
    masks0 = [x[0] for x in masks0]  
    preds_df['pred0'] = masks0
    
    masks = Parallel(n_jobs)(delayed(modify_w_unet)(preds_df.loc[ind, 'pred0'],
                     np.concatenate(preds_df.loc[ind, ['mask','ly','lx','ldr','ldl']], -1))
                for ind in preds_df.index)
    preds_df['pred'] = masks

    save_to_cache(preds_df, os.path.join(weight_dir, df_name))


def get_cons_local_valid(mask, preds):
    nb_mask = mask.max()
    assert len(np.unique(mask))==nb_mask+1
    assert preds.shape[2]==5
    
    mask_s, y1, y2, x1, x2 = trans2square(mask)
    local_true = mask_localpos(mask_s)[y1:mask_s.shape[0]-y2, x1:mask_s.shape[1]-x2]
    local_pred = preds[:,:,1:]
    #local_pred = np.concatenate(preds[['y_true','x_true','dr_true','dl_true']],2)
    #local_pred = np.concatenate(preds[['y','x','dr','dl']],2)    
    scores = np.zeros(nb_mask, dtype='float32')
    for nb in range(nb_mask):
        index = mask==nb+1
        score = np.mean((local_pred[index]-local_true[index])**2)
        scores[nb] = score
        
    return scores

def get_cons(df_name, weight_dir, n_jobs = 18):    
    fl_name = os.path.join(weight_dir, df_name)
    preds_df = load_from_cache(fl_name)
    print(preds_df.shape)
    cons_total = Parallel(n_jobs)(delayed(get_cons_local_valid)(preds_df.loc[ind, 'pred'],
                         np.concatenate(preds_df.loc[ind, ['mask','ly','lx','ldr','ldl']], -1))
                            for ind in preds_df.index)
    preds_df['con'] = cons_total
    save_to_cache(preds_df, os.path.join(weight_dir, '{}'.format(df_name)))



def get_cons_scale(df_name, weight_dir, tag='half', n_jobs=18):
    fl_name = os.path.join(weight_dir, '{}_{}'.format(df_name, tag))
    if tag!='two':
        preds_df = load_from_cache(fl_name)
    else:
        fl_names = glob.glob(os.path.join(weight_dir, '{}_{}'.format(df_name, tag),
                                          '{}_{}_[0-9+].dat'.format(df_name, tag)))+\
                   glob.glob(os.path.join(weight_dir, '{}_{}'.format(df_name, tag), 
                                          '{}_{}_[0-9][0-9].dat'.format(df_name, tag)))
        preds_df = load_from_cache_multi(os.path.join(weight_dir, 
                                                      '{}_{}'.format(df_name, tag),
                                                      '{}_{}'.format(df_name, tag)), 
                   nb=len(fl_names))
    print(preds_df.shape)
    cons_total = Parallel(n_jobs)(delayed(get_cons_local_valid)(preds_df.loc[ind, 'pred_scale'],
                         np.concatenate(preds_df.loc[ind, ['mask','ly','lx','ldr','ldl']], -1))
                            for ind in preds_df.index)
    preds_df['con'] = cons_total
    if tag!='two':
        save_to_cache(preds_df, os.path.join(weight_dir, '{}_{}'.format(df_name, tag)))
    else:
        save_to_cache_multi(preds_df, os.path.join(weight_dir, 
                                                   '{}_{}'.format(df_name, tag),
                                                   '{}_{}'.format(df_name, tag)),10)    
def get_mask_scale(df_name, config, weight_dir, save_name, tag='half', n_jobs=18):
    fl_name = os.path.join(weight_dir, df_name+'_'+tag)
    if os.path.exists('{}.dat'.format(fl_name)):
        preds_df = load_from_cache(fl_name)
    else:
        fl_names = glob.glob(os.path.join(weight_dir, '{}_{}_[0-9+].dat'.format(df_name, tag)))+\
                   glob.glob(os.path.join(weight_dir, '{}_{}_[0-9][0-9].dat'.format(df_name, tag)))
        
        preds_df = load_from_cache_multi(fl_name, nb=len(fl_names))
    
    print(preds_df.shape)
    
    save_dir = os.path.join(weight_dir, save_name)
    os.path.exists(save_dir)
    df = load_from_cache(df_name)
    
    masks0_scale = Parallel(n_jobs)(delayed(postprocess)(np.concatenate(preds_df.loc[ind, 
                      ['mask','ly','lx','ldr','ldl']], -1), config) for ind in preds_df.index)
    masks0_scale = Parallel(n_jobs)(delayed(renumber_mask)(mask) for mask in masks0_scale)
    preds_df['pred0_scale'] = [x[0].astype('int16') for x in masks0_scale]  

    masks_scale = Parallel(n_jobs)(delayed(modify_w_unet)(preds_df.loc[ind, 'pred0_scale'],
                     np.concatenate(preds_df.loc[ind, ['mask','ly','lx','ldr','ldl']], -1))
                    for ind in preds_df.index)
    preds_df['pred_scale'] = [x.astype('int16') for x in masks_scale]
    
    masks = Parallel(n_jobs)(delayed(cv2.resize)(preds_df.loc[ind, 'pred_scale'],
                      dsize=(df.loc[ind, 'shape'][1], df.loc[ind, 'shape'][0]),
                      interpolation=cv2.INTER_NEAREST) for ind in df.index)
    preds_df['pred'] = [x.astype('int16') for x in masks]
    
    if tag!='two':
        save_to_cache(preds_df, os.path.join(weight_dir, '{}_{}'.format(df_name, tag)))
    else:
        save_to_cache_multi(preds_df, os.path.join(weight_dir, '{}_{}'.format(df_name, tag),
                                             '{}_{}'.format(df_name, tag)),10)
        


    


