"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os, re, glob
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
#import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from keras.callbacks import Callback

from lib import my_bacc, my_brec, weighted_mae, my_mae2
import utils
from load import MetricsLogger
# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from model import inst_weight

def test():
    from params import DsbConfig
    config = DsbConfig()
    model = UnetRCNN('all', 0.1, 'sgd', config, '..//cache')
    
    filepath = '/home/work/dsb/cache/mask_rcnn_coco.h5'
    model.load_weights(filepath, by_name=True)

class cbk(Callback):
    def __init__(self, fold_dir):
        self.fold_dir = fold_dir
    def on_epoch_end(self, epoch, logs=None):
        weights = glob.glob(os.path.join(self.fold_dir, '*.hdf5'))
        if len(weights)>1:
            weights = sorted(weights, key = lambda x: float(x[:-5].split('_')[-1]))[1:]
            for weight in weights:
                os.remove(weight)
        val_loss = logs['val_loss']
        if val_loss>6:
            self.model.stop_training=True
    
class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def C1_graph(input_image, ksize=1, depth=1):
    x = input_image
    for nb_depth in range(depth):
        x = KL.Conv2D(256, (ksize, ksize), name='head_C1_{}_conv1'.format(nb_depth), use_bias=True, 
                      padding='same')(x)
        x = BatchNorm(axis=3, name='head_C1_{}_bn_conv1'.format(nb_depth))(x)
        x = KL.Activation('relu')(x)
    return x
############################################################
#  MaskRCNN Class
############################################################

class UnetRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, tp, unet_ratio, opt, config, fold_dir, img_size= None):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert tp in ['di', 'all', 'xy']
        self.tp = tp
        self.config = config
        self.opt = opt
        self.fold_dir = fold_dir
        self.img_size = config.IMAGE_SHAPE[0] if img_size is None else img_size
        self.logger = MetricsLogger(os.path.join(fold_dir, 'log.txt'))
        self.unet_ratio = unet_ratio
        self.nb_start = 2 if tp=='di' else 0
        self.nb_feature = 8 if tp=='all' else 2
        self.keras_model = self.build(tp=tp, config=config)


    def build(self, tp, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        # Image size must be dividable by 2 multiple times
        if self.img_size / 2**6 != int(self.img_size / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=(self.img_size*4, self.img_size*4, 3),name='input_image')
        weight_true = KL.Input(shape=(self.img_size, self.img_size, 1))
        #input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        #input_up = KL.Conv2DTranspose(3, (7,7), strides=(4,4), padding='same',
        #                              name = 'head_input_convt')(input_image)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
     
        #C1 = C1_graph(input_image, ksize=config.C1_KSIZE, depth=config.C1_DEPTH)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        '''
        for nb_p1 in range(2):    
            P2 = KL.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation="relu",
                               name="head_fpn_p2ct_{}".format(nb_p1))(P2)
        P1 = KL.Add(name="head_fpn_p1add")([P2,
            KL.Conv2D(256, (1, 1), name = 'head_fpn_c1p1')(C1)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        
        P1 = KL.Conv2D(256, (1, 1), padding="SAME", activation='relu',
                       name="head_fpn_p1")(P1)
        '''
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        
        P2_true = KL.Conv2D(256, (3, 3), padding="SAME", activation='relu',
                            name="fpn_p2_true_conv0")(P2)          
        output_y_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_y_true_conv')(P2_true)
        output_x_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_x_true_conv')(P2_true)
        output_dr_true= KL.Conv2D(1, (1, 1), activation='tanh', name='output_dr_true_conv')(P2_true)
        output_dl_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_dl_true_conv')(P2_true)

        weight_pred = KL.Lambda(lambda x: inst_weight(*x, config = config), name="weight_pred")(
                    [output_y_true, output_x_true, output_dr_true, output_dl_true])
        weight = KL.Maximum()([weight_pred, weight_true])

        output_y_true = KL.Concatenate(name='output_y_true')([output_y_true, weight_true])
        output_x_true = KL.Concatenate(name='output_x_true')([output_x_true, weight_true])
        output_dr_true= KL.Concatenate(name='output_dr_true')([output_dr_true, weight_true])
        output_dl_true = KL.Concatenate(name='output_dl_true')([output_dl_true, weight_true])
        
        P2_pred = KL.Conv2D(256, (3, 3), padding="SAME", activation='relu',
                            name="fpn_p2_pred_conv0")(P2)            
        output_mask = KL.Conv2D(1, (1, 1), activation='sigmoid', 
                                name='output_mask')(P2_pred)    
        output_y = KL.Conv2D(1, (1, 1), activation='tanh', name='output_y_conv')(P2_pred)
        output_x = KL.Conv2D(1, (1, 1), activation='tanh', name='output_x_conv')(P2_pred)
        output_dr= KL.Conv2D(1, (1, 1), activation='tanh', name='output_dr_conv')(P2_pred)
        output_dl = KL.Conv2D(1, (1, 1), activation='tanh', name='output_dl_conv')(P2_pred)
        output_ly = KL.Conv2D(1, (1, 1), activation='tanh', name='output_ly_conv')(P2_pred)
        output_lx = KL.Conv2D(1, (1, 1), activation='tanh', name='output_lx_conv')(P2_pred)
        output_ldr= KL.Conv2D(1, (1, 1), activation='tanh', name='output_ldr_conv')(P2_pred)
        output_ldl = KL.Conv2D(1, (1, 1), activation='tanh', name='output_ldl_conv')(P2_pred)

        output_y = KL.Concatenate(name='output_y')([output_y, weight])
        output_x = KL.Concatenate(name='output_x')([output_x, weight])
        output_dr= KL.Concatenate(name='output_dr')([output_dr, weight])
        output_dl = KL.Concatenate(name='output_dl')([output_dl, weight])
        output_ly = KL.Concatenate(name='output_ly')([output_ly, weight])
        output_lx = KL.Concatenate(name='output_lx')([output_lx, weight])
        output_ldr= KL.Concatenate(name='output_ldr')([output_ldr, weight])
        output_ldl = KL.Concatenate(name='output_ldl')([output_ldl, weight])
        
        output_inst = [output_y_true, output_x_true, output_dr_true, output_dl_true, 
                       output_y, output_x, output_dr, output_dl, 
                       output_ly,output_lx,output_ldr,output_ldl]
        #output_inst =[output_ly, output_lx, output_ldr, output_ldl][self.nb_start:
         #               self.nb_start+self.nb_feature]
    
        model = KM.Model(inputs=[input_image, weight_true], 
                         outputs=[output_mask]+ output_inst)

        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        return model

    def compile(self, learning_rate):
        output_names = self.keras_model.output_names
        loss, metrics, loss_weights = {},{},{}
        for name in output_names:
            if 'mask' in name:
                loss[name] = 'binary_crossentropy'
                metrics[name]=[my_bacc, my_brec]
                loss_weights[name] = self.unet_ratio
            else:
                loss[name] = weighted_mae
                metrics[name] = [my_mae2]
                loss_weights[name] = 1

        if self.opt=='adam':
            opt = keras.optimizers.Adam(learning_rate)
        elif self.opt =='sgd':
            opt = keras.optimizers.SGD(lr=learning_rate, 
                                       momentum=self.config.LEARNING_MOMENTUM,
                                       clipnorm=5.0)
        self.keras_model.compile(optimizer=opt, loss=loss, 
                      metrics = metrics, 
                      loss_weights = loss_weights)           
        
    def load_weights(self, filepath, by_name=False, exclude=None, verbose=1):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology
        
        if verbose==1:
            self.logger.log('loading weights form {}'.format(filepath))
        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            self.logger.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                self.logger.log("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                self.logger.log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))


    def train_generator(self, train_generator, val_generator, learning_rate, epochs, layers, clbcks=[]):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "head": r"(output\_.*)|(head\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.fold_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(os.path.join(self.fold_dir, '{epoch:2d}_'+'{val_loss:.4f}.hdf5'), 
                                            monitor='val_loss', save_best_only = True, 
                                            verbose=1, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1), 
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=3,
                                              verbose = 1),
            #cbk(self.fold_dir)                                  
        ] + clbcks 

        # Train
        self.logger.log("\nLR={}\n".format(learning_rate))
        self.set_trainable(layers, verbose = 1)
        self.compile(learning_rate)
        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = 2
            #max(self.config.BATCH_SIZE // 2, 1)

        self.keras_model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=400,
            workers=workers,
            use_multiprocessing=False,
        )


    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        windows = np.stack(windows)
        return molded_images, windows

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rois, rpn_class, rpn_bbox =\
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
