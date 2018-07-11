# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:45:21 2018

@author: Jackie
"""
import numpy as np
import pandas as pd

from load import load_from_cache, save_to_cache, makefolder
from model_ms1 import generator_1s_v11 as generator_1s
from model_rcnn_weight import UnetRCNN
from params import DsbConfig
from generator import data_generator_multi

def train_generator(config, shuffle = False, augment=False):
    isbi = load_from_cache('2009isbi_256')
    weebly = load_from_cache('weebly_256')
    tnbc = load_from_cache('TNBC_256')
    train = load_from_cache('train_final_df')
    gen_isbi = data_generator_multi(np.stack(isbi['image'],0), np.stack(isbi['mask'], 0),
                                    config, shuffle=shuffle, augment=augment, batch_size=1,
                                    tp_value=7)
    gen_weebly = data_generator_multi(np.stack(weebly['image'],0), np.stack(weebly['mask'], 0),
                                    config, shuffle=shuffle, augment=augment, batch_size=1)    
    gen_tnbc = data_generator_multi(np.stack(tnbc['image'],0), np.stack(tnbc['mask'], 0),
                                config, shuffle=shuffle, augment=augment, batch_size=1)
    gen_train = data_generator_multi(np.stack(train['image'],0), np.stack(train['mask'], 0),
                                    config, shuffle=shuffle, augment=augment, batch_size=1)
    images = np.zeros((config.BATCH_SIZE,)+tuple(config.IMAGE_SHAPE), dtype='float32')
    masks = np.zeros((config.BATCH_SIZE,)+tuple(config.IMAGE_SHAPE[:2]), dtype='int32')
    while True:
        for nb, gen in enumerate([gen_isbi, gen_weebly, gen_tnbc, gen_train]):
            image, mask = next(gen)
            images[nb] = image[0]
            masks[nb] = mask[0]
        yield images, masks
        
def train_1s():
    config = DsbConfig()
    tp, unet_ratio, opt = 'all', 1, 'sgd'
    save_dir = makefolder('..//cache//UnetRCNN',tosecond=True)
    weight_fl = '../cache/mask_rcnn_coco.h5'

    valid_df = load_from_cache('valid_df')
    valid_images = np.stack(valid_df['image'], 0)
    valid_masks = np.stack(valid_df['mask'], 0)
    #print(len(valid_masks))
    
    model = UnetRCNN(tp, unet_ratio, opt, config, save_dir)
    model.load_weights(weight_fl, by_name =True)
    train_gen = train_generator(config, shuffle = True, augment= True)
    tr_ms1 = generator_1s(train_gen,config, tp)
    val_generator = data_generator_multi(valid_images, valid_masks, 
                                         config, shuffle = False, augment= False)        
    val_ms1= generator_1s(val_generator,config, tp)
        
    #model.train_generator(tr_ms1, val_ms1, 1e-2, 1, 'head')
    model.train_generator(tr_ms1, val_ms1, 1e-3, 100, 'all')
 
if __name__ == "__main__":   
    train_1s()
        

    



