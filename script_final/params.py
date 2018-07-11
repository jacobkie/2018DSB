#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:14:25 2018

@author: work
"""
from config import Config
import numpy as np
#import os
train_dir = '..//input//train'
test_dir = '..//input//test'
save_dir = '..//cache//train'
stage2_dir = '..//input//stage2_test_final'

class DsbConfig(Config):

    # Give the configuration a recognizable name
    NAME = "dsb"
      
    LEARNING_RATE = 1e-2
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution image
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Train on 1 GPU and 8 images per GPU. Batch size is GPUs * images/GPU.
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1
    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
    # typically be equal to the number of samples of your dataset divided by the batch size
    STEPS_PER_EPOCH = 1010
    VALIDATION_STEPS = 505

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nucleis
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_PADDING = True  # currently, the False option is not supported
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels, maybe add a 256?
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    RPN_ANCHOR_RATIOS = [np.asarray([1,1,np.sqrt(2), np.sqrt(2)]),
                         #np.asarray([1,1,2,2]),
                         #np.asarray([1,1,1,1])*np.sqrt(2),
                         np.asarray([1/2,1,3/2,3/2])*np.sqrt(2),
                         np.asarray([3/2,3/2,1,2]),
                         np.asarray([1,1/2,3/2,3/2])*np.sqrt(2),
                         np.asarray([3/2,3/2,2,1])]

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 500 #300 nb of rpn bbox, at most 150 be positive (rpn_match==1)
    
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 256    ###max instance train in detection_target layer(train only)
    RPN_NMS_THRESHOLD = 0.7
    MAX_GT_INSTANCES = 256 #####256
    DETECTION_MAX_INSTANCES = 256 ###max instance predict in detection layer (predict only)
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3 # 0.3
    
    
    MEAN_PIXEL = np.array([0.,0.,0.])
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    # affine transform
    ROTATION_RANGE = 10
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    SHEAR_RANGE = 10
    ZOOM_RANGE = 1.2
    
    # elastic transform
    ALPHA = 200
    SIGMA = 10

    #white_noise
    NOISE_RATIO = 0.1
    
    #blur
    BLUR_KSIZE = 3
    BLUR_SIGMA = 10
    #mask threshold
    MASK_THRESHOLD = 5
    
    C1_DEPTH = 2
    C1_KSIZE = 3
    
    WEIGHT_AREA = 100
    TOUCH_RATIO = 1
    BG_AREA = 1500
    CLIP_AREA_HIGH = 1500
    CLIP_AREA_LOW = 30
    GRADIENT_THRES = 0.01
    MASK_THRES=0.1
    WALL_DEPTH = 0.5
    GAUSSIAN_KERNEL = np.asarray([[0.0392052 , 0.03979772, 0.0399972 , 0.03979772, 0.0392052 ],
                                 [0.03979772, 0.04039918, 0.04060168, 0.04039918, 0.03979772],
                                 [0.0399972 , 0.04060168, 0.0408052 , 0.04060168, 0.0399972 ],
                                 [0.03979772, 0.04039918, 0.04060168, 0.04039918, 0.03979772],
                                 [0.0392052 , 0.03979772, 0.0399972 , 0.03979772, 0.0392052 ]]).reshape((5,5,1,1))
