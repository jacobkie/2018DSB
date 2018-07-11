#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 05:24:19 2018

@author: work
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 07:53:43 2018

@author: work
"""
import numpy as np

from preprocess import affine_transform_batch, elastic_transform_batch,\
        enhance, level_noise, stretch, insert

def num_iterator(total, batch_size, shuffle=True):
    nb_seen = 0
    inds = np.arange(total)
    if shuffle:
        np.random.shuffle(inds)
    
    while True:
        if nb_seen + batch_size > total:
            ind_batch = inds[nb_seen:]
            if shuffle:
                np.random.shuffle(inds)    
            nb_seen = nb_seen + batch_size - total
            ind_batch = np.hstack([ind_batch, inds[:nb_seen]])
        else:
            ind_batch = inds[nb_seen:nb_seen+batch_size]
            nb_seen += batch_size
        yield ind_batch


def data_generator_multi(images, masks, config, shuffle=True, augment=False, 
                         batch_size=None, tp_value = 8):
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    iterator = num_iterator(masks.shape[0], batch_size, shuffle = shuffle)
    
    while True:
        ind_batch = next(iterator)
        image_batch = images[ind_batch].copy().astype('float32')
        mask_batch = masks[ind_batch].copy()
        if augment:
            tp = np.random.randint(tp_value)
            if tp==1:
                if np.random.rand()>0.5:
                    image_batch = image_batch[:,:,::-1]
                    mask_batch = mask_batch[:,:,::-1]
                else:
                    image_batch = image_batch[:,::-1]
                    mask_batch = mask_batch[:,::-1]                
            elif tp==2:
                image_batch, mask_batch = affine_transform_batch(image_batch, mask_batch, 
                                        rotation_range = config.ROTATION_RANGE,
                                        width_shift_range = config.WIDTH_SHIFT_RANGE,
                                        height_shift_range = config.HEIGHT_SHIFT_RANGE,
                                        shear_range = config.SHEAR_RANGE,
                                        zoom_range = config.ZOOM_RANGE)
            elif tp==3:
                image_batch, mask_batch = elastic_transform_batch(image_batch, mask_batch,
                                        alpha = config.ALPHA, sigma = config.SIGMA)

            elif tp==4:
                tries = np.random.randint(20,100)
                max_fails = 5
                image_batch, mask_batch = insert(image_batch, mask_batch, tries, max_fails)
                
            elif tp==5:
                image_batch = enhance(image_batch, mask_batch, max_tries=10, max_enhance_ratio=0.8)
            elif tp==6:
                image_batch = level_noise(image_batch, mask_batch, max_level_ratio=4,
                                          max_noise_ratio=0.05)
            elif tp==7:
                image_batch, mask_batch = stretch(image_batch, mask_batch, max_ratio=2)
        
        '''
        image_max = np.max(image_batch, axis=(1,2,3), keepdims=True)
        image_max[image_max==0]=1
        amax = np.random.randint(200, 256) if augment else 255
        image_batch = image_batch/image_max*amax
        '''
        yield [image_batch.astype('float32'), mask_batch]

 
