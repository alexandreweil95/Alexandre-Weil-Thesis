# This file contains all the models used in this project. 


import sys
import time
from random import shuffle
import numpy as np
import argparse
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.layers import Input, Activation, GlobalAveragePooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
# from .metrics import bag_accuracy, bag_loss, binary_focal_loss
# from .custom_layers import Mil_Attention, Last_Sigmoid



def simple_conv_net(input_dim, args, weights=None, useMulGpu=False): # Simple ConvNet trained as baseline https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    
    conv1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))

    
    embed= GlobalAveragePooling2D()(conv2)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    
    x_mul = multiply([alpha, embed], name='multiply_1')
    # x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
    
    model = Model(inputs=[data_input], outputs=[out])



    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model






def instance_based_net(input_dim, args, weights=None, useMulGpu=False): # Model a) in Ilse et. al (2018). Based on Mi-Net : https://github.com/yanyongluan/MINNs with added convolutional layers

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    pool1 = MaxPooling2D((2,2))(conv1_2)
    conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    pool2 = MaxPooling2D((2,2))(conv2_2)
    conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    pool3 = MaxPooling2D((2,2))(conv3_3)
    conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    pool4 = MaxPooling2D((2,2))(conv4_3)
    conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
    embed= GlobalAveragePooling2D()(conv5_3)

    sp = Score_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='lse', name='sp')(embed) # FCN then MIL (like mi-Net)

    model = Model(inputs=[data_input], outputs=[sp])

    ## Old approach for the last few layers:

    # fcn = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(embed) # FCN 
    # fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max', name='fp')(fcn) # MIL
    # model = Model(inputs=[data_input], outputs=[fp])

    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model





def embedding_based_net(input_dim, args, weights=None, useMulGpu=False): # Model b) in Ilse et. al (2018). Based on Mi-Net : https://github.com/yanyongluan/MINNs with added convolutional layers

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    
#     conv1_2 = Dropout(0.5)(conv1_2) # Dropout layer added
    
    pool1 = MaxPooling2D((2,2))(conv1_2)
    conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    
#     conv2_2 = Dropout(0.5)(conv2_2) # Dropout layer added
    
    pool2 = MaxPooling2D((2,2))(conv2_2)
    conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    
#     conv3_3 = Dropout(0.5)(conv3_3) # Dropout layer added
    
    pool3 = MaxPooling2D((2,2))(conv3_3)
    conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    
#     conv4_3 = Dropout(0.5)(conv4_3) # Dropout layer added
    
    pool4 = MaxPooling2D((2,2))(conv4_3)
    conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
    
#     conv5_3 = Dropout(0.5)(conv5_3) # Dropout layer added
    
    embed= GlobalAveragePooling2D()(conv5_3)

    fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='lse', name='fp')(embed) # MIL Pooling then FC (like in MI-Net)

    model = Model(inputs=[data_input], outputs=[fp])
    

    ## Old approach for the last few layers:

    # fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max', name='fp')(embed) # MIL Pooling + FC
    # out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(fp) # FCN
    # model = Model(inputs=[data_input], outputs=[out])
    

    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model






def covid_net(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) (with attention mechanism as the MIL pooling)

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    
    conv1_2_Dropout = Dropout(0.9)(conv1_2) # Dropout layer added
    
    
    pool1 = MaxPooling2D((2,2))(conv1_2_Dropout)
    conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    
    
    conv2_2_Dropout = Dropout(0.9)(conv2_2) # Dropout layer added
    
    
    pool2 = MaxPooling2D((2,2))(conv2_2_Dropout)
    conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    

    conv3_3_Dropout = Dropout(0.9)(conv3_3) # Dropout layer added
    
    
    pool3 = MaxPooling2D((2,2))(conv3_3_Dropout)
    conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    
    conv4_3_Dropout = Dropout(0.9)(conv4_3) # Dropout layer added
    
    
    pool4 = MaxPooling2D((2,2))(conv4_3_Dropout)
    conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
        
    conv5_3_Dropout = Dropout(0.9)(conv5_3) # Dropout layer added

    embed= GlobalAveragePooling2D()(conv5_3_Dropout)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    
    x_mul = multiply([alpha, embed], name='multiply_1')
    # x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
    
    model = Model(inputs=[data_input], outputs=[out])



    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model




###### Additional backbones below ######


#import tensorflow as tf
#from tensorflow import keras

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import keras.layers
from keras.layers import ZeroPadding2D, Add, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply


from tensorflow.keras import layers # Added
from keras.layers import Layer # Added

import os
import collections
import warnings



# from .. import get_submodules_from_kwargs
# from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None



def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
    Args:
        reduction: channels squeeze factor
    """
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channels_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = tf.keras.backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = GlobalAveragePooling2D()(x)
        x = Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = Activation('sigmoid')(x)

        # apply attention
        x = Multiply()([input_tensor, x])

        return x

    return layer


ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'attention']
)


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])

        return x

    return layer


# ResNet18

import tensorflow as tf
from tensorflow import keras

MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block, None),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block, None),
    'resnet50': ModelParams('resnet50', (3, 4, 6, 3), residual_bottleneck_block, None),
    'resnet101': ModelParams('resnet101', (3, 4, 23, 3), residual_bottleneck_block, None),
    'resnet152': ModelParams('resnet152', (3, 8, 36, 3), residual_bottleneck_block, None),
}


model_params = MODELS_PARAMS['resnet18'] # Can change this line to change version of ResNet


def covid_ResNet18(input_dim, args, input_shape=None, input_tensor=None, weights=None, include_top=False, useMulGpu=False): # Model c) in Ilse et. al (2018) with ResNet18 backbone instead of the standard CNN backbone

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]
    
#     # choose residual block type
#     ResidualBlock = model_params.residual_block
#     if model_params.attention:
#         Attention = model_params.attention(**kwargs)
#     else:
#         Attention = None

    classes = 2
    Attention = None


    # if tf.keras.backend.image_data_format() == 'channels_last': # Added tf.keras.
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    # if input_tensor is None:
    #     data_input = Input(shape=input_shape, dtype='float32', name='input')
    # else:
    #     if not tf.keras.backend.is_keras_tensor(input_tensor):
    #         data_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         data_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(data_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='post', attention=Attention)(x)

            elif block == 0:
                x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                  cut='post', attention=Attention)(x)

            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='pre', attention=Attention)(x)

    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

#     # resnet top
#     if include_top:
#         x = GlobalAveragePooling2D(name='pool1')(x)
#         x = Dense(classes, name='fc1')(x)
#         x = Activation('softmax', name='softmax')(x)


    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model



import keras.layers
from keras.layers import ZeroPadding2D, Add, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply



from tensorflow.keras import layers # Added
from keras.layers import Layer # Added

import os
import warnings

backend = None
layers = None
models = None
keras_utils = None


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    if tf.keras.backend.image_data_format() == 'channels_last': # Added tf.keras.
        bn_axis = 3
    else:
        bn_axis = 1

    # bn_axis = 3 # [HACK] manually added since the image_data_format is 'channels_last'

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor]) # changed from layers.add
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    if tf.keras.backend.image_data_format() == 'channels_last': # Added tf.keras.
        bn_axis = 3
    else:
        bn_axis = 1

    # bn_axis = 3 # [HACK] manually added since the image_data_format is 'channels_last'

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut]) # Changed from layers.add
    x = Activation('relu')(x)
    return x

# ResNet50
#import tensorflow.keras

# (ResNet is simplified version of Densenet - densenet works very well for segmentation tasks but here we work on image level classification so Resnet is more helpful than DenseNet)

# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
#from tensorflow.keras.applications.resnet import ResNet50

import tensorflow as tf
from tensorflow import keras

def covid_ResNet50(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) with ResNet50 backbone instead of the standard CNN backbone

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if tf.keras.backend.image_data_format() == 'channels_last': # Added tf.keras.
        bn_axis = 3
    else:
        bn_axis = 1

    # bn_axis = 3 # [HACK] manually added

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    # conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    # conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    # pool1 = MaxPooling2D((2,2))(conv1_2)
    # conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    # conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    # pool2 = MaxPooling2D((2,2))(conv2_2)
    # conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    # conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    # conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    # pool3 = MaxPooling2D((2,2))(conv3_3)
    # conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    # conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    # conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    # pool4 = MaxPooling2D((2,2))(conv4_3)
    # conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    # conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    # conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
    
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(data_input)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
  
    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model



# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


from keras import backend
import keras.layers
from keras.layers import Concatenate, concatenate

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    
    if tf.keras.backend.image_data_format() == 'channels_first': # Added tf.keras.
        bn_axis = 1
    else:
        bn_axis = 3

    # bn_axis = 3 # [HACK] manually added since the image_data_format is 'channels_last'


    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x



# Inception v3

from tensorflow.keras.applications.inception_v3 import InceptionV3

def covid_InceptionV3(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) with InceptionV3 backbone instead of the standard CNN backbone

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]


    if tf.keras.backend.image_data_format() == 'channels_first': # Added tf.keras.
        channel_axis = 1
    else:
        channel_axis = 3
        
    data_input = Input(shape=input_dim, dtype='float32', name='input')

    x = conv2d_bn(data_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model





### This is for SE ResNet50

try:
    from tensorflow import __version__ as tf_version

    TF = True
except ImportError:
    TF = False

def _tensor_shape(tensor):
    return getattr(tensor, '_shape_val') if TF else getattr(tensor, '_keras_shape')


import keras.layers
from keras.layers import Reshape, add, Permute, Conv2D, Concatenate, ZeroPadding2D, Add, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply

# def _tensor_shape(tensor):
#     return getattr(tensor, '_keras_shape') # Modified since using only keras.layers
#     # return getattr(tensor, 'tf_keras_shape')
#     # return getattr(tensor, 'shape')

def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if tf.keras.backend.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _conv2d_bn(x,
               filters,
               num_row,
               num_col,
               padding='same',
               strides=(1, 1),
               name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input keras tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = '{name}_bn'.format(name=name)
        conv_name = '{name}_conv'.format(name=name)
    else:
        bn_name = None
        conv_name = None


    bn_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x




def _resnet_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block without bottleneck layers
    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block with bottleneck layers
    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


# def _create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
#                       depth, width, bottleneck, weight_decay, pooling):
def covid_SqueezeExcite_ResNet50(img_input, args, weights=None, bottleneck=True, useMulGpu=False): # Model c) in Ilse et. al (2018) with Squeeze-Excite ResNet50 backbone instead of the standard CNN backbone
    """Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    """

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    depth = [3, 4, 6, 3] # ResNet50
    filters = [64, 128, 256, 512] # Default
    assert len(depth) == len(filters) # "The length of filter increment list must match the length " \"of the depth list."

    width =1 # Default, can increase to get Wide ResNet
    initial_conv_filters = 32 # Same as VGG backbone
    classes = 3 # COVID, CAP, Normal

    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(data_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model



# DenseNet

import keras.layers
from keras.layers import Concatenate, ZeroPadding2D, Add, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, # change to tf.keras.backend?
                      use_bias=False,
                      name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_d_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_d_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def covid_DenseNet(blocks, input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) with DenseNet backbone instead of the standard CNN backbone

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(data_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)


    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])

    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model






# Squeeze and Excite Inception-ResNet V2

import keras.layers
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D


# Squeeze and Excite block already defined above

def conv2d_bn_InceptionResNetV2(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input keras tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else '{name}_bn'.format(name=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else '{name}_ac'.format(name=name)
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block with Squeeze and Excitation block at the end.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input keras tensor.
        scale_: scaling factor to scale_ the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale_ * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn_InceptionResNetV2(x, 32, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(x, 32, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 32, 3)
        branch_2 = conv2d_bn_InceptionResNetV2(x, 32, 1)
        branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 48, 3)
        branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn_InceptionResNetV2(x, 192, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(x, 128, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn_InceptionResNetV2(x, 192, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(x, 192, 1)
        branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: {block_type}'.format(block_type=block_type))

    block_name = '{block_type}_{block_idx}'.format(block_type=block_type, block_idx=block_idx)
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name='{block_name}_mixed'.format(block_name=block_name))(branches)
    up = conv2d_bn_InceptionResNetV2(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name='{block_name}_conv'.format(block_name=block_name))

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up]) # Bug comes from here: TypeError: <lambda>() got an unexpected keyword argument 'scale'

    # x = Lambda(lambda inputs, scale_: inputs[0] + inputs[1] * scale_,
    #            output_shape=K.int_shape(x)[1:],
    #            arguments={'scale': scale},
    #            name=block_name)([x, up]) # Bug comes from here: TypeError: <lambda>() got an unexpected keyword argument 'scale'


    if activation is not None:
        x = Activation(activation, name='{block_name}_ac'.format(block_name=block_name))(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)
    return x



def covid_SEInceptionResNetV2(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) with Squeeze Excite InceptionResNetV2 backbone instead of the standard CNN backbone
    """Instantiates the SE-Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.
    The model and the weights are compatible with both TensorFlow and Theano
    backends (but not CNTK). The data format convention used by the model is
    the one specified in your Keras config file.
    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or `'imagenet'` (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with an unsupported backend.
    """
    
    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    classes = 2 # Covid, Non-Covid (binary classification)

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    # Stem block: 35 x 35 x 192
    x = conv2d_bn_InceptionResNetV2(data_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn_InceptionResNetV2(x, 32, 3, padding='valid')
    x = conv2d_bn_InceptionResNetV2(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn_InceptionResNetV2(x, 80, 1, padding='valid')
    x = conv2d_bn_InceptionResNetV2(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn_InceptionResNetV2(x, 96, 1)
    branch_1 = conv2d_bn_InceptionResNetV2(x, 48, 1)
    branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 64, 5)
    branch_2 = conv2d_bn_InceptionResNetV2(x, 64, 1)
    branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 96, 3)
    branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn_InceptionResNetV2(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn_InceptionResNetV2(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_InceptionResNetV2(x, 256, 1)
    branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 256, 3)
    branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn_InceptionResNetV2(x, 256, 1)
    branch_0 = conv2d_bn_InceptionResNetV2(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_InceptionResNetV2(x, 256, 1)
    branch_1 = conv2d_bn_InceptionResNetV2(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn_InceptionResNetV2(x, 256, 1)
    branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 288, 3)
    branch_2 = conv2d_bn_InceptionResNetV2(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn_InceptionResNetV2(x, 1536, 1, name='conv_7b')


    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model
        
    return parallel_model





# Squeeze and Excite Inception V3

import keras.layers
from keras.layers import Reshape, add, Permute, Conv2D, Concatenate, ZeroPadding2D, Add, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply

# def _tensor_shape(tensor):
#     return getattr(tensor, '_keras_shape') # Modified since using only keras.layers
#     # return getattr(tensor, 'tf_keras_shape')
#     # return getattr(tensor, 'shape')

def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if tf.keras.backend.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _conv2d_bn(x,
               filters,
               num_row,
               num_col,
               padding='same',
               strides=(1, 1),
               name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input keras tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = '{name}_bn'.format(name=name)
        conv_name = '{name}_conv'.format(name=name)
    else:
        bn_name = None
        conv_name = None


    bn_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def covid_SqueezeExcite_InceptionV3(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) with Squeeze-Excite InceptionV3 backbone instead of the standard CNN backbone

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
    data_input = Input(shape=input_dim, dtype='float32', name='input')

    x = _conv2d_bn(data_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = _conv2d_bn(x, 32, 3, 3, padding='valid')
    x = _conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv2d_bn(x, 80, 1, 1, padding='valid')
    x = _conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 1: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 3: 17 x 17 x 768
    branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 4: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 128, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 160, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed{}'.format(5 + i))

        # squeeze and excite block
        x = squeeze_excite_block(x)

    # mixed 7: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 192, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = _conv2d_bn(x, 192, 1, 1)
    branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3,
                           strides=(2, 2), padding='valid')

    branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = _conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 320, 1, 1)

        branch3x3 = _conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_{i}'.format(i=i))

        branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed{}'.format(9 + i))

        # squeeze and excite block
        x = squeeze_excite_block(x)

    embed= GlobalAveragePooling2D()(x)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(embed) # using the defined MIL attention class here
    x_mul = multiply([alpha, embed])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,), precision_m, recall_m])
        parallel_model = model

    return parallel_model


##### End of additional backbones section #####
