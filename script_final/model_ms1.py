# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:32:22 2018

@author: Jackie
"""

import numpy as np
import cv2
from scipy.signal import convolve2d

from mask_pos import mask_relpos, mask_localpos, get_labels, get_coords


def get_mask_weights(mask, config):
    mask_weight = np.ones(mask.shape, dtype='float32')*config.WEIGHT_AREA/config.BG_AREA    
    index = mask!=0

    vals, counts = np.unique(mask[index], return_counts = True)
    inds = np.zeros(mask.max()+1, dtype='int32')
    inds[vals] = np.clip(counts, config.CLIP_AREA_LOW, config.CLIP_AREA_HIGH)
    mask_weight[index] = config.WEIGHT_AREA/inds[mask[index]]
    mask_weight = convolve2d(mask_weight, config.GAUSSIAN_KERNEL[:,:,0,0], 'same')
    return mask_weight

        
def generator_1s_v11(gen, config, tp='all'):
    nb_feature = 8 if tp=='all' else 4
    mask_pos = [np.zeros((config.BATCH_SIZE, config.IMAGE_SHAPE[0], 
                          config.IMAGE_SHAPE[0], 1), dtype='float32') 
                for nb in range(nb_feature)]
    mask_weight = np.zeros((config.BATCH_SIZE, config.IMAGE_SHAPE[0],
                            config.IMAGE_SHAPE[0], 1), dtype='float32')
    inp = np.zeros((config.BATCH_SIZE, 1024, 1024, 3), dtype='float32')
    labels0 = get_labels(config.IMAGE_SHAPE[0])
    coords = get_coords(config.IMAGE_SHAPE[0])
    while True:
        mask_pos = [x*0 for x in mask_pos]
        inp = inp*0
        mask_weight = mask_weight*0
        
        image, mask = next(gen)
        mask_unet = (mask>0).reshape(mask.shape+(1,)).astype('int32')
        for nb in range(config.BATCH_SIZE):
            mask_weight[nb, :,:,0] = get_mask_weights(mask[nb,:,:], config)
            res = mask_relpos(mask[nb], tp)
            for nb_f in range(nb_feature//2):
                mask_pos[nb_f][nb] = res[:,:,nb_f:nb_f+1]
            res = mask_localpos(mask[nb], tp=tp, labels0=labels0, coords = coords)
            for nb_en, nb_f in enumerate(range(nb_feature//2, nb_feature)):
                mask_pos[nb_f][nb] = res[:,:, nb_en:nb_en+1]
            inp[nb] = cv2.resize(image[nb], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        yield [inp, mask_weight], [mask_unet]+ mask_pos[:4] + mask_pos