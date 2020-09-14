
 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# Select GPU to use - The GPU ids are: "0", "1" or "2" or "3";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

# Hack to make other Keras backbones work
import tensorflow as tf
import tensorflow.keras.backend as tfback


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus




from keras import backend as K

def max_pooling(x):
    """Max Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do max-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of max-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.max(x, axis=0, keepdims=True)
    return output

def mean_pooling(x):
    """Mean Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do mean-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of mean-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.mean(x, axis=0, keepdims=True)
    return output

def LSE_pooling(x):
    """LSE Pooling to obtain aggregation.
    Do LSE(log-sum-exp) pooling, like LSE(x1, x2, x3) = log(exp(x1)+exp(x2)+exp(x3)).
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do LSE-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of LSE-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.log(K.mean(K.exp(x), axis=0, keepdims=True))
    return output

def choice_pooling(x, pooling_mode):
    """Choice the pooling mode
    Parameters
    -------------------
    x : Tensor (N x d)
        Input data to do MIL-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    pooling_mode : string
        Choice the pooling mode for MIL pooling.
    Return
    --------------------
    output : Tensor (1 x d)
            Output of MIL-pooling,
            where d is dimension of instance feature
            (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    if pooling_mode == 'max':
        return max_pooling(x)
    if pooling_mode == 'lse':
        return LSE_pooling(x)
    if pooling_mode == 'ave':
        return mean_pooling(x)




from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers, regularizers
#from . import pooling_method as pooling



class Score_pooling(Layer): # For model a). Source: https://github.com/yanyongluan/MINNs/blob/f816645f5d7bbc9366b95fc967f775426fc081c1/mil_nets/layer.py#L45
    """
    Score pooling layer
    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.
    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, pooling_mode='ave', **kwargs):
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Score_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape

        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        x = K.sigmoid(x)

        # do-pooling operator
        output = choice_pooling(x, self.pooling_mode)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias,
            'pooling_mode': self.pooling_mode
        }
        base_config = super(Score_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






class Feature_pooling(Layer): # For model b). Source: https://github.com/yanyongluan/MINNs/blob/f816645f5d7bbc9366b95fc967f775426fc081c1/mil_nets/layer.py#L45
    """
    Feature pooling layer
    This layer contains a MIL pooling and a FC layer which only has one neural with
    sigmoid activation. The input of this layer is instance features. Via MIL pooling,
    we aggregate instance features to bag features. Finally, we obtain bag score by
    this FC layer with only one neural and sigmoid activation
    This layer is used in MI-Net and MI-Net with DS.
    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, pooling_mode='max', **kwargs):
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Feature_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        trainable=True)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator
        x = choice_pooling(x, self.pooling_mode)

        # compute bag-level score
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        # sigmoid
        output = K.sigmoid(output)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias,
            'pooling_mode': self.pooling_mode
        }
        base_config = super(Feature_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers, regularizers

class Mil_Attention(Layer):
    """
    Mil Attention Mechanism
    This layer contains Mil Attention Mechanism
    # Input Shape
        2D tensor with shape: (batch_size, input_dim)
    # Output Shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=False, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.tanh(K.dot(x, self.V)) # (2,64)

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(ori_x, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        alpha = K.softmax(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Last_Sigmoid(Layer): # Needed for model c)
    """
    Attention Activation
    This layer contains a FC layer which only has one neural with sigmoid activation
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.
    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





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

    sp = Score_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='ave', name='sp')(embed) # FCN then MIL (like mi-Net)

    model = Model(inputs=[data_input], outputs=[sp])

    ## Old approach for the last few layers:

    # fcn = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(embed) # FCN 
    # fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max', name='fp')(fcn) # MIL
    # model = Model(inputs=[data_input], outputs=[fp])

    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [binary_focal_loss(gamma=2., alpha=.25)], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [binary_focal_loss(gamma=2., alpha=.25)], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
        parallel_model = model

    return parallel_model




def embedding_based_net(input_dim, args, weights=None, useMulGpu=False): # Model b) in Ilse et. al (2018). Based on Mi-Net : https://github.com/yanyongluan/MINNs with added convolutional layers

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    
#     conv1_2_Dropout = Dropout(0.5)(conv1_2) # Dropout layer added
    
    pool1 = MaxPooling2D((2,2))(conv1_2)
    conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    
#     conv2_2_Dropout = Dropout(0.5)(conv2_2) # Dropout layer added
    
    pool2 = MaxPooling2D((2,2))(conv2_2)
    conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    
#     conv3_3_Dropout = Dropout(0.5)(conv3_3) # Dropout layer added
    
    pool3 = MaxPooling2D((2,2))(conv3_3)
    conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    
    conv4_3_Dropout = Dropout(0.5)(conv4_3) # Dropout layer added
    
    pool4 = MaxPooling2D((2,2))(conv4_3_Dropout)
    conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
    
    conv5_3_Dropout = Dropout(0.5)(conv5_3) # Dropout layer added
    
    embed= GlobalAveragePooling2D()(conv5_3_Dropout)

    fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max', name='fp')(embed) # MIL Pooling then FC (like in MI-Net)

    model = Model(inputs=[data_input], outputs=[fp])
    


    ## Old approach for the last few layers:

    # fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max', name='fp')(embed) # MIL Pooling + FC
    # out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(fp) # FCN
    # model = Model(inputs=[data_input], outputs=[out])
    

    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [binary_focal_loss(gamma=2., alpha=.25)], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [binary_focal_loss(gamma=2., alpha=.25)], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
        parallel_model = model

    return parallel_model



def covid_net(input_dim, args, weights=None, useMulGpu=False): # Model c) in Ilse et. al (2018) (with attention mechanism as the MIL pooling)

    lr = args["init_lr"]
    weight_decay = args["init_lr"]
    momentum = args["momentum"]

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1_1 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(data_input)))
    conv1_2 = Activation('relu')(BatchNormalization()(Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv1_1)))
    
#     conv1_2_Dropout = Dropout(0.9)(conv1_2) # Dropout layer added
    
    
    pool1 = MaxPooling2D((2,2))(conv1_2)
    conv2_1 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool1)))
    conv2_2 = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv2_1)))
    
    
#     conv2_2_Dropout = Dropout(0.9)(conv2_2) # Dropout layer added
    
    
    pool2 = MaxPooling2D((2,2))(conv2_2)
    conv3_1 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool2)))
    conv3_2 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_1)))
    conv3_3 = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv3_2)))
    

#     conv3_3_Dropout = Dropout(0.9)(conv3_3) # Dropout layer added
    
    
    pool3 = MaxPooling2D((2,2))(conv3_3)
    conv4_1 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool3)))
    conv4_2 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_1)))
    conv4_3 = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv4_2)))
    
#     conv4_3_Dropout = Dropout(0.9)(conv4_3) # Dropout layer added
    
    
    pool4 = MaxPooling2D((2,2))(conv4_3)
    conv5_1 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(pool4)))
    conv5_2 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_1)))
    conv5_3 = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3,3), kernel_regularizer=l2(weight_decay))(conv5_2)))
        
#     conv5_3_Dropout = Dropout(0.9)(conv5_3) # Dropout layer added

    embed= GlobalAveragePooling2D()(conv5_3)

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
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
        parallel_model = model

    return parallel_model




###### Add additional backbones here ######


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
    
    x_mul = multiply([alpha, embed], name='multiply_1')

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

    model = Model(inputs=[data_input], outputs=[out])


    # model.summary()
    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss= [bag_loss], metrics=[bag_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity_m, tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None,)])
        parallel_model = model

    return parallel_model


##### End of additional backbones section #####




import numpy as np
import random
import threading
# from .data_aug_op import random_flip_img, random_rotate_img
#from keras.preprocessing.image import ImageDataGenerator
import scipy.misc as sci

class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataGenerator(object):
    def __init__(self, batch_size=32, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __Get_exploration_order(self, list_patient, shuffle):
        indexes = np.arange(len(list_patient))
        if shuffle:
            random.shuffle(indexes)
        return indexes

    def __Data_Genaration(self, batch_train):
        bag_batch = []
        bag_label = []
        
        #datagen = ImageDataGenerator()
                                                                       
        #transforms = {
        	#	"theta" : 0.25,
        	#	"tx" : 0.2,
        	#	"ty" : 0.2,
        	#	"shear" : 0.2,
        	#	"zx" : 0.2,
        	#	"zy" : 0.2,
        #   "flip_horizontal" : True,
        #		"zx" : 0.2,
        	#	"zy" : 0.2,
        	#}
        
        for ibatch, batch in enumerate(batch_train):
            aug_batch = []
            img_data = batch[0]
            for i in range(img_data.shape[0]):
                ori_img = img_data[i, :, :, :]
                # sci.imshow(ori_img)
                if self.shuffle:
                    img = random_flip_img(ori_img, horizontal_chance=0.5, vertical_chance=0.5)
                    img = np.expand_dims(random_rotate_img(img), 2)
                    #img = datagen.apply_transform(ori_img,transforms)
                else:
                    img = ori_img
                exp_img = np.expand_dims(img, 0)
                # sci.imshow(img)
                aug_batch.append(exp_img)
            input_batch = np.concatenate(aug_batch, 0)
            bag_batch.append((input_batch))
            bag_label.append(batch[1])

        return bag_batch, bag_label


    def generate(self, train_set):
        flag_train = self.shuffle

        while 1:

        # status_list = np.zeros(batch_size)
        # status_list = []
            indexes = self.__Get_exploration_order(train_set, shuffle=flag_train)

        # Generate batches
            imax = int(len(indexes) / self.batch_size)

            for i in range(imax):
                Batch_train_set = [train_set[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                X, y = self.__Data_Genaration(Batch_train_set)

                yield X, y


        # batch_train_set = Generate_Batch_Set(train_set, batch_size, flag_train)  # Get small batch from the original set


        # print img_list_1[0].shape
        # yield img_list, status_list





import random
import numpy as np
import cv2

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res

def random_rotate_img(images):
    rand_roat = np.random.randint(4, size=1)
    angle = 90*rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst

def random_crop(image, crop_size=(400, 400)):
    height, width = image.shape[:-1]
    dy, dx = crop_size
    X = np.copy(image)
    aX = np.zeros(tuple([3, 400, 400]))
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    aX = X[y:(y + dy), x:(x + dx), :]
    return aX





destination_folder= '/app/Alex/Images_Destination_Folder'  # [for UCL Cluster]
# # destination_folder= '/content/drive/My Drive/Thesis (Aladdin)/Images Destination Folder'  # [for Google Colab]





import numpy as np
import glob
from sklearn.model_selection import KFold

def load_dataset(dataset_path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    # load datapath from path
    pos_path = glob.glob(dataset_path+'/1/*/*/*')
    neg_path = glob.glob(dataset_path+'/0/*/*/*')

    pos_num = len(pos_path)
    neg_num = len(neg_path)
    
    print('The number of positive paths is:',pos_num)
    print('The number of negative paths is:',neg_num)
    
    all_path = pos_path + neg_path

    # num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path): # This is throwing a bug
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
#         print('dataset["train"] is equal to:',dataset['train'][0:3])x
#         print('dataset["test"] is equal to:',dataset['test'][0:3])
        
        datasets.append(dataset)
    return datasets




from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import clip_ops

epsilon = backend_config.epsilon

def _constant_to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.
  Arguments:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)

def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(target, output):
        target = K.mean(target, axis=0, keepdims=False)
        output = K.mean(output, axis=0, keepdims=False)
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or output.op.type != 'Sigmoid'):
            epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
            output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

        return -K.mean(target* alpha * K.pow(1. - output + epsilon(), gamma) * K.log(output + epsilon())) \
               -K.mean((1-target)*(1 - alpha) * K.pow(output, gamma) * K.log(1. - output + epsilon()))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=1)

    return categorical_focal_loss_fixed



    # To implement more comprehensive metrics: 
  # Precision, Recall (most important), Sensitivity, Specificity, AUC (plot ROC curves=> very important metric to see how good classifier is). 
  # Should plot AUC across different number of training bags. 
  # No need for the F1

# https://keras.io/api/metrics/
# Plot AUC across different number of training bags: [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py]

import sklearn
from sklearn import metrics
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score


# Sensitivity (measures the proportion of actual positives that are correctly identified as such (tp / (tp + fn))) https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.metrics.sensitivity_score.html

def sensitivity_m(y_true, y_pred):

    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Specificity (measures the proportion of actual negatives that are correctly identified as such (tn / (tn + fp)))

def specificity_m(y_true, y_pred):

    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)

    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Plot ROC
def plot_roc(name, labels, predictions, **kwargs): 
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


# plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
# plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
# plt.legend(loc='lower right')


def plot_roc_curve(y_true, y_pred):
  plot_roc("Train Baseline", y_true, y_pred)
  # plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
  plt.legend(loc='lower right')




# Added to allow training of models with other backbones

def _get_available_gpus():
  """Get a list of available gpu devices (formatted as strings).

  # Returns
    A list of available GPU devices.
  """
  global _LOCAL_DEVICES
  if _LOCAL_DEVICES is None:
      if _is_tf_1():
          devices = get_session().list_devices()
          _LOCAL_DEVICES = [x.name for x in devices]
      else:
          devices = tf.config.list_logical_devices()
          _LOCAL_DEVICES = [x.name for x in devices]
      return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]



#!/usr/bin/env python
'''
This is a re-implementation of the following paper:
"Attention-based Deep Multiple Instance Learning"
I got very similar results but some data augmentation techniques not used here
https://128.84.21.199/pdf/1802.04712.pdf
*---- Jiawen Yao--------------*
'''

import numpy as np
import time
#from utl import Covid_Net
from random import shuffle
import argparse
from keras.models import Model
#from utl.dataset import load_dataset
#from utl.data_aug_op import random_flip_img, random_rotate_img
import glob
import imageio
import tensorflow as tf
from PIL import Image 

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from scipy.special import softmax

import matplotlib.pyplot as plt

import os

model_saving_folder='/app/Alex/model_saving_folder'
if not os.path.exists(model_saving_folder):
        os.mkdir(model_saving_folder)


def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    # parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    # parser.add_argument('--lr', dest='init_lr',
    #                     help='initial learning rate',
    #                     default=1e-4, type=float)
    # parser.add_argument('--decay', dest='weight_decay',
    #                     help='weight decay',
    #                     default=0.0005, type=float)
    # parser.add_argument('--momentum', dest='momentum',
    #                     help='momentum',
    #                     default=0.9, type=float)
    # parser.add_argument('--epoch', dest='max_epoch',
    #                     help='number of epoch to train',
    #                     default=100, type=int)
    # parser.add_argument('--useGated', dest='useGated',
    #                     help='use Gated Attention',
    #                     default=False, type=int)
    # parser.add_argument('--train_bags_samples', dest='train_bags_samples',
    #                     help='path to save sampled training bags',
    #                     default='./save_train_bags', type=str)
    # parser.add_argument('--test_bags_samples', dest='test_bags_samples',
    #                     help='path to save test bags',
    #                     default='./test_results', type=str)
    # parser.add_argument('--model_id', dest='model_id',
    #                     help='path to model',
    #                     default='./child_only', type=str)

    # # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    

    model_id = model_saving_folder + '/Covid_Net_ResNet18'
    

    args = dict(init_lr = 1e-4,
    weight_decay = 0.0005,
    momentum = 0.9,
    max_epoch=50, # Keep modify it depending on the model and how quickly it overfits
    useGated = True, # Change this to get gated or not                                 
    train_bags_samples = model_saving_folder + '/save_train_bags',
    test_bags_samples = model_saving_folder + '/test_results',
    model_id = model_id,
    test_visualisation = model_id + '/test_results_Visualisation_outputs' + '/Show_intra') # Need to modify this since we changed dataset
 
    if not os.path.exists(args["model_id"]):
        os.mkdir(args["model_id"])
    if not os.path.exists(args["test_visualisation"]):
        os.mkdir(args["test_visualisation"])        
    if not os.path.exists(args["train_bags_samples"]):
        os.mkdir(args["train_bags_samples"])
    if not os.path.exists(args["test_bags_samples"]):
        os.mkdir(args["test_bags_samples"])
    if not os.path.exists(args["model_id"] + "/Saved_model"):
        os.mkdir(args["model_id"] + "/Saved_model")
    if not os.path.exists(args["model_id"] + "/Results/"):
        os.mkdir(args["model_id"] + "/Results/") # Created to save graph of training and validation loss over number of epochs   
    return args

def generate_batch(path, mode=None):
    bags = []
    num_pos= 0
    num_neg= 0 
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.png')
        img_names = [ i.rsplit('/')[-1] for i in img_path]
        img_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        img_path = [ each_path + '/' + i for i in img_names]
#         label = int(each_path.split('/')[-2]) # Problematic line
        label = int(each_path.split('/')[-4]) # new line
        
#         print('int(each_path.split('/')[-2])=',int(each_path.split('/')[-2]))
#         print('int(each_path.split('/'))=',int(each_path.split('/')))
#         print('label=',label)
        
        if not img_path:
          continue

        if mode== 'train':

            for each_img in img_path[0:len(img_path):int(np.ceil(0.025*len(img_path)))]:
                img_data = Image.open(each_img)
                #img_data -= 255
                img_data = img_data.resize((224,224),Image.BILINEAR)


                img_data =np.array(img_data, dtype='float32')
                if len(np.shape(img_data)) != 2:
                    img_data= img_data[:,:,0]
                else:
                    img_data= img_data
                img_data= (img_data-img_data.mean()) / img_data.std()
                img.append(np.expand_dims(np.expand_dims(img_data,2),0))
                name_img.append(each_img.split('/')[-1])
            if label == 1:
                curr_label = np.ones(len(img), dtype=np.uint8) # deprecated
#                 curr_label = np.ones(len(img), dtype=object) # Use this one instead?
                num_pos += 1
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8) # deprecated
#                 curr_label = np.zeros(len(img), dtype=object) # Use this one instead?                             
                num_neg += 1
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))

            img= []
            name_img =[]
            for each_img in img_path[0:len(img_path):int(np.ceil(0.05 * len(img_path)))]:
                img_data = Image.open(each_img)
                #img_data -= 255
                img_data = img_data.resize((224,224),Image.BILINEAR)
                img_data =np.array(img_data, dtype='float32')
                if len(np.shape(img_data)) != 2:
                    img_data= img_data[:,:,0]
                else:
                    img_data= img_data
                img_data= (img_data-img_data.mean()) / img_data.std()
                img.append(np.expand_dims(np.expand_dims(img_data,2),0))
                name_img.append(each_img.split('/')[-1])
            if label == 1:
                curr_label = np.ones(len(img), dtype=np.uint8)
                num_pos += 1 
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8)
                num_neg += 1 
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))

            img = []
            name_img = []
            for each_img in img_path[0:len(img_path):int(np.ceil(0.075 * len(img_path)))]:
                img_data = Image.open(each_img)
                # img_data -= 255
                img_data = img_data.resize((224, 224), Image.BILINEAR)
                img_data = np.array(img_data, dtype='float32')
                if len(np.shape(img_data)) != 2:
                    img_data = img_data[:, :, 0]
                else:
                    img_data = img_data
                img_data = (img_data - img_data.mean()) / img_data.std()
                img.append(np.expand_dims(np.expand_dims(img_data, 2), 0))
                name_img.append(each_img.split('/')[-1])
            if label == 1:
                curr_label = np.ones(len(img), dtype=np.uint8)
                num_pos += 1
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8)
                num_neg += 1
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))

            img = []
            name_img = []
            for each_img in img_path[0:len(img_path):int(np.ceil(0.1 * len(img_path)))]:
                img_data = Image.open(each_img)
                # img_data -= 255
                img_data = img_data.resize((224, 224), Image.BILINEAR)
                img_data = np.array(img_data, dtype='float32')
                if len(np.shape(img_data)) != 2:
                    img_data = img_data[:, :, 0]
                else:
                    img_data = img_data
                img_data = (img_data - img_data.mean()) / img_data.std()
                img.append(np.expand_dims(np.expand_dims(img_data, 2), 0))
                name_img.append(each_img.split('/')[-1])
            if label == 1:
                curr_label = np.ones(len(img), dtype=np.uint8)
                num_pos += 1
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8)
                num_neg += 1
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))

        else:

            img = []
            name_img = []
            for each_img in img_path[0:len(img_path):int(np.ceil(0.01 * len(img_path)))]:
                img_data = Image.open(each_img)
                # img_data -= 255
                img_data = img_data.resize((224, 224), Image.BILINEAR)
                img_data = np.array(img_data, dtype='float32')
                if len(np.shape(img_data)) != 2:
                    img_data = img_data[:, :, 0]
                else:
                    img_data = img_data
                img_data = (img_data - img_data.mean()) / img_data.std()
                img.append(np.expand_dims(np.expand_dims(img_data, 2), 0))
                name_img.append(each_img.split('/')[-1])
            if label == 1:
                curr_label = np.ones(len(img), dtype=np.uint8)
                num_pos += 1
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8)
                num_neg += 1
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))


    return bags, num_pos, num_neg


def Get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    import random
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """

    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)

    ## Added for the new metrics
    test_precision = np.zeros((num_test_batch, 1), dtype=float)
    test_recall = np.zeros((num_test_batch, 1), dtype=float)
    test_specificity = np.zeros((num_test_batch, 1), dtype=float)
    test_AUC = np.zeros((num_test_batch, 1), dtype=float)


    for ibatch, batch in enumerate(test_set): # Throwing tracing error      
        
#         print('x = batch[0].shape = ',batch[0].shape) # To help debug
#         print('y = batch[1].shape = ',batch[1].shape) # To help debug
#         print('y = batch[1] = ',batch[1]) # To help debug

#         result = model.test_on_batch(x=batch[0], y=batch[1]) # Problematic- Throwing an error
        result = model.test_on_batch(x=batch[0], y=batch[1][:1]) # y=batch[1] is a vector of all ones or all zeros for the bag. Take first element to have the right dimension to compare to y_true

#         result = model.test_on_batch(x=batch[0], y=np.mean(batch[1],keepdims=True))
        
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]

#         print('In test_eval, len(result) = ',len(result)) # This is a list of 7 items. len(result)=7
        test_precision[ibatch] = result[2]
        test_recall[ibatch] = result[3]
        test_specificity[ibatch] = result[4]
        test_AUC[ibatch] = result[5]

#     y_preds = np.concatenate([model.predict(batch[0]).ravel() for batch in test_set]) # added
#     y_trues = np.concatenate([batch[1] for batch in test_set]) # added
#     plot_roc_curve(y_trues, y_preds) # added

    return np.mean(test_loss), np.mean(test_acc), np.mean(test_precision), np.mean(test_recall), np.mean(test_specificity), np.mean(test_AUC)


def train_eval(model, train_set, irun, ifold):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set, train_percentage=0.9)



    # from utl.DataGenerator import DataGenerator
    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    model_name = args["model_id"]+"/Saved_model/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + "best.hd5"
    # model_name = args["model_id"]+"/Saved_model/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + "best.tf"


    # Hack to save Mobile Net V3 Model without a bug
    # a = Input(shape=(10,))
    # out = tf.tile(a, (1, tf.shape(a)[0]))
    # model_name = Model(a, out)




#     checkpoint_fixed_name = ModelCheckpoint(model_name,
#                                             monitor='val_loss', verbose=1, save_best_only=True,
#                                             save_weights_only=True, mode='auto', period=1) # Working fine but deprecated

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', save_freq='epoch')

    
    

    EarlyStop = EarlyStopping(monitor='val_loss', patience=8)

    callbacks = [checkpoint_fixed_name, EarlyStop]


    ### Useful read: https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    # history = model.fit_generator(generator=train_gen, steps_per_epoch=len(model_train_set)//batch_size,
    #                                          epochs=args["max_epoch"], validation_data=val_gen,
    #                                         validation_steps=len(model_val_set)//batch_size, callbacks=callbacks) # Deprecated. Use Please use Model.fit, which supports generators.

    history = model.fit(x=train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             epochs=args["max_epoch"], validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks) # Replaces model.fit_generator


    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = args["model_id"] +'/Results/' + str(irun) + '_' + str(ifold) + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)


    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = args["model_id"]+'/Results/' + str(irun) + '_' + str(ifold) + "_val_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    return model_name


def model_training(input_dim, dataset, irun, ifold):

    train_bags = dataset['train']
    test_bags = dataset['test']
    # convert bag to batch
    train_set, num_pos, num_neg = generate_batch(train_bags, mode='train')
    test_set, _, _ = generate_batch(test_bags, mode='test')
    num_bags= num_pos+num_neg
    
    print('This is what num_bags looks like:',num_bags) # To debug
    print('This is what num_pos looks like:',num_pos) # To debug
    
    inv_freq= np.array([num_bags/num_pos, num_bags/num_neg], dtype='float32')
    normalised_inv_freq= softmax(inv_freq) # added global
    
    dirc= args["model_id"]+'/data'
    if not os.path.exists(dirc):
        os.mkdir(dirc)
#     np.save(os.path.join(dirc, 'fold_{}_train_bags.npy'.format(ifold)), train_set) # [27th July - Commented out since not needed]
    np.save(os.path.join(dirc, 'fold_{}_test_bags.npy'.format(ifold)), test_set) # Modify this for n-fold training?
    #fig, ax = plt.subplots()
    #ax.set_axis_off()
    #for ibatch, batch in enumerate(train_set):
     #   dirs = os.path.join(args.train_bags_samples, str(ibatch))
    #    if not os.path.exists(dirs):
    #        os.mkdir(dirs)
    #    for im in range(batch[0].shape[0]):
     #       img = np.squeeze(batch[0][im], 2)
     #       plt.imshow(img, cmap='gray', interpolation='bilinear')
     #       extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #        plt.savefig(os.path.join(dirs, batch[2][im]), bbox_inches=extent1)


    ### Beginning of model selection panel ###

#     model = instance_based_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model a)
#     model = embedding_based_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model b)
#     model = covid_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c)

    model = covid_ResNet18(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with ResNet18 backbone
    

#     model = covid_ResNet50(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with ResNet50 backbone
    # model = covid_InceptionV3(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with InceptionV3 backbone
    

    # model = covid_SqueezeExcite_ResNet50(input_dim, args, normalised_inv_freq, bottleneck=True, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze-Excite ResNet50 backbone instead of the standard CNN backbone

    # model = covid_SqueezeExcite_InceptionV3(input_dim, args, normalised_inv_freq, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze-Excite InceptionV3 backbone instead of the standard CNN backbone


    ## Test out the below:
    # model = covid_SEInceptionResNetV2(input_dim, args, normalised_inv_freq, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze Excite InceptionResNetV2 backbone instead of the standard CNN backbone
    

    ## Pick DenseNet version
    blocks = [6, 12, 24, 16] # DenseNet121 Model
    # blocks = [6, 12, 32, 32] # DenseNet169 Model
    # blocks = [6, 12, 48, 32] # DenseNet201 Model
    # blocks = [TO BE SPECIFIED BY USER] # DenseNet Model
    
    # model = covid_DenseNet(blocks, input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with DenseNet backbone

    
    ### End of model selection panel ### 


    # train model
    t1 = time.time()
    # for epoch in range(args.max_epoch):

    model_name = train_eval(model, train_set, irun, ifold)

    #print("load saved model weights")
    #model.load_weights(model_name)

    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy
        
    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)


    #t2 = time.time()
    #out_test = open('Results/' + 'test_results.txt', 'w')
    #out_test.write("fold{} run_time:{:.3f} min  test_acc:{:.3f} ".format(ifold, (t2 - t1) / 60.0, test_acc,))
    #out_test.write("\n")

    return model_name

print('Model is now ready to train!')




import numpy as np
from keras.models import Model
import os


# def parse_args():
#     """Parse input arguments.
#     Parameters
#     -------------------
#     No parameters.
#     Returns
#     -------------------
#     args: argparser.Namespace class object
#         An argparse.Namespace class object contains experimental hyper-parameters.
#     """
#     parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
#     parser.add_argument('--lr', dest='init_lr',
#                         help='initial learning rate',
#                         default=1e-4, type=float)
#     parser.add_argument('--decay', dest='weight_decay',
#                         help='weight decay',
#                         default=0.0005, type=float)
#     parser.add_argument('--momentum', dest='momentum',
#                         help='momentum',
#                         default=0.9, type=float)
#     parser.add_argument('--epoch', dest='max_epoch',
#                         help='number of epoch to train',
#                         default=100, type=int)
#     parser.add_argument('--useGated', dest='useGated',
#                         help='use Gated Attention',
#                         default=False, type=int)
#     parser.add_argument('--train_bags_samples', dest='train_bags_samples',
#                         help='path to save sampled training bags',
#                         default='./save_train_bags', type=str)
#     parser.add_argument('--test_bags_samples', dest='test_bags_samples',
#                         help='path to save test bags',
#                         default='./test_results', type=str)

#     # if len(sys.argv) == 1:
#     #     parser.print_help()
#     #     sys.exit(1)



#     ### Below section has been added to the code [Remove?? ]
#     args = dict(init_lr = 1e-4, 
#     weight_decay = 0.0005,
#     momentum = 0.9,
#     max_epoch=50, # Keep on 100 for experiments
#     useGated = True, 
# #     train_bags_samples = destination_folder + '/save_train_bags_Visualisation',
# #     test_bags_samples = destination_folder + '/test_results_Visualisation',
#     train_bags_samples = model_saving_folder + '/save_train_bags_Visualisation',
#     test_bags_samples = model_saving_folder + '/test_results_Visualisation_outputs',
# #      os.path.exists(args["model_id"]
                
#     model_id_visualisation = model_saving_folder +'/Visualisation') 

#     if not os.path.exists(args["model_id_visualisation"]):
#         os.mkdir(args["model_id_visualisation"])
#     if not os.path.exists(args["train_bags_samples"]):
#         os.mkdir(args["train_bags_samples"])
#     if not os.path.exists(args["test_bags_samples"]):
#         os.mkdir(args["test_bags_samples"])
#     if not os.path.exists(args["model_id"] + "/Saved_model"):
#         os.mkdir(args["model_id"] + "/Saved_model")

#     ### End of added section
    
#     return args




def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """

    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)

    ## Added for the new metrics
    test_precision = np.zeros((num_test_batch, 1), dtype=float)
    test_recall = np.zeros((num_test_batch, 1), dtype=float)
    test_specificity = np.zeros((num_test_batch, 1), dtype=float)
    test_AUC = np.zeros((num_test_batch, 1), dtype=float)


    for ibatch, batch in enumerate(test_set): # Throwing tracing error      
        
#         print('x = batch[0].shape = ',batch[0].shape) # To help debug
#         print('y = batch[1].shape = ',batch[1].shape) # To help debug
#         print('y = batch[1] = ',batch[1]) # To help debug

#         result = model.test_on_batch(x=batch[0], y=batch[1]) # Problematic- Throwing an error
        result = model.test_on_batch(x=batch[0], y=batch[1][:1]) # y=batch[1] is a vector of all ones or all zeros for the bag. Take first element to have the right dimension to compare to y_true

#         result = model.test_on_batch(x=batch[0], y=np.mean(batch[1],keepdims=True))
        
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]

#         print('In test_eval, len(result) = ',len(result))
        test_precision[ibatch] = result[2]
        test_recall[ibatch] = result[3]
        test_specificity[ibatch] = result[4]
        test_AUC[ibatch] = result[5]

#     y_preds = np.concatenate([model.predict(batch[0]).ravel() for batch in test_set]) # added
#     y_trues = np.concatenate([batch[1] for batch in test_set]) # added
#     plot_roc_curve(y_trues, y_preds) # added

    return np.mean(test_loss), np.mean(test_acc), np.mean(test_precision), np.mean(test_recall), np.mean(test_specificity), np.mean(test_AUC)



args = parse_args()
# print('Called with args:')
# print(args)

input_dim = (224,224,1)

# test_set = np.load('/content/drive/My Drive/Thesis (Aladdin)/Images Destination Folder/child_only/data/fold_4_test_bags.npy', allow_pickle=True) # To modify



test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify

checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'

# checkpoint_path = args["model_id"]+"/Saved_model/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + ".ckpt"


    
model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone


model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load

test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

print('test_loss:%.3f' %test_loss)
print('test_acc:%.3f' %test_acc)
print('test_precision:%.3f' %test_precision)
print('test_recall:%.3f' %test_recall)
print('test_specificity:%.3f' %test_specificity)
print('test_AUC:%.3f' %test_AUC)




### Imports for Visualisation

import numpy as np
from keras.models import Model
# from utl import Covid_Net
import argparse
import matplotlib.pyplot as plt
import os
from keras.utils import plot_model
from keras import backend as K
from scipy.special import softmax

from keras.layers import Layer
from keras import activations, initializers, regularizers

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy import ndimage, misc


#### Start of one_plot.py ####

def show_images(images, cols = 1, titles = None, ibatch=None, batch=None, intermediate_output=None, score=None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    # dirs = os.path.join(args.test_bags_samples, str(ibatch)) # change
    dirs = os.path.join(args["test_visualisation"], str(ibatch))

    labels= ['Non-COVID', 'COVID']
    label = np.mean(batch[1], axis=0, keepdims=False)
    if int(label)== 1:
        dirs= dirs + '_'+ labels[int(label)]+':%.2f' %score +'.png'
    else:
        dirs= dirs + '_'+ labels[int(label)]+':%.2f' %(1-score) +'.png'
    
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    plt.clf()
    plt.cla()
    plt.close()
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)): # change?
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title('weight'+':%.6f' %np.around(intermediate_output[n], 6), fontsize=100)
        a.set_axis_off()
        #a.text(70, 12, labels[batch[1][n]]+':%.6f' %np.around(intermediate_output[n], 6), color='green', fontsize=100)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    #plt.set_title(labels[batch[1][0]]+':%.2f' %score, fontsize=128)
    plt.savefig(dirs)
    

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    input_dim = (224,224,1)

    test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify

    checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'

        
    model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone

    model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load
   
    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)
    

    # model_outputs = Model(inputs=model.input,
    #                                  outputs=[model.get_layer('FC1_sigmoid').output, model.get_layer('model_1').get_layer('alpha').output])
    
    model_outputs = Model(inputs=model.input,
                                     outputs=[model.get_layer('FC1_sigmoid').output, model.get_layer('alpha').output])
    
    labels = ['Non-COVID', 'COVID']
    for ibatch, batch in enumerate(test_set):
        
        score, intermediate_output = model_outputs.predict_on_batch(x=batch[0]) # intermediate_output is weighted sum of feature vectors

        images = []
        for im in range(batch[0].shape[0]):
            img = np.squeeze(batch[0][im], 2)
            images.append(img)
        show_images(images, cols = 4, ibatch= ibatch, batch=batch, intermediate_output=intermediate_output, score=np.mean(score, axis=0, keepdims=False))

# Try this code on the Aladdin GPU and see if there is enough RAM


#### End of one_plot.py ####




#### Start of "patient_specific_clustering.py" ####

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    input_dim = (224,224,1)

    test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify
    checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'
    
    model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone

    
    model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load
    
    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)
    
    
    model_outputs = Model(inputs=model.input,
                                     outputs=[model.get_layer('multiply_1').output, model.get_layer('alpha').output])
    
    
    target_names= ['g', 'y', 'r'] # not used
    colors = ['g', 'y', 'r'] # not used
    target_ids = range(len(target_names))
    
    
    
#     ### Debugging section
#     print('start of debugging')
    
#     count_Covid = 0
#     count_Non_Covid = 0
#     count_non_classified_images = 0
    
#     for ibatch, batch in enumerate(test_set):
# #         print('print(batch[1])=',batch[1]) # to debug
        
#         if np.mean(batch[1], axis=0, keepdims=False)==1:
#             count_Covid += 1
#         elif np.mean(batch[1], axis=0, keepdims=False)==0:
#             count_Non_Covid += 1
#         else:
#             count_non_classified_images += 1
        
#     print('The number of Covid images in this test set is:',count_Covid)
#     print('The number of NON Covid images in this test set is:',count_Non_Covid)
#     print('The number of unclassified images in this test set is:',count_non_classified_images)
    
#     print('end of debugging')
#     ### Debugging section
    
    
    for ibatch, batch in reversed(list(enumerate(test_set))): # reversed() to go around for loop in opposite direction

#     for ibatch, batch in enumerate(test_set): 
        
        # print(batch[0])
        ins_emb, weights = model_outputs.predict_on_batch(x=batch[0])
        
        
        weights= np.squeeze(np.around(weights, decimals=4),1)
        
#         predicted_label = model_outputs.predict_on_batch(x=batch[0])
#         print("predicted_label=",predicted_label)
#         print("len(predicted_label)=",len(predicted_label))
        
        
        # dirs = os.path.join(args.test_bags_samples, str(ibatch)) # Modify dirs
        dirs = os.path.join(args["test_visualisation"], str(ibatch))

        
        # Adaptive Thresholding
        Top_10 = np.percentile(weights, 90)
        Bottom_10 = np.percentile(weights, 10)

        labels  = np.asarray([2 if i >= Top_10 else 1 if Top_10>i>= Bottom_10 else 0 for i in weights], dtype= 'int64') # Added
                
        
        ## No need to use K-means
      
        # kmeans = KMeans(2)
        # labels = kmeans.fit_predict(ins_emb)

        tsne = TSNE(n_components=2, perplexity=5)
        tsne_2d = tsne.fit_transform(ins_emb)

        ## No need to use PCA
        # pca = PCA(2)  # project from 64 to 2 dimensions
        # projected = pca.fit_transform(ins_emb)
        
      
        targets= ['NON-COVID', 'COVID']
        # label_names= ['weights<1%', 'weights>=1%']
        label_names= ['Bottom 10%', 'Middle 80%', 'Top 10%'] # Added

        label_ids = range(len(label_names))
        target = np.mean(batch[1], axis=0, keepdims=False) # batch[1] is the true label y of the bag
    
    
        if int(target)== 1:
            histogram_dirs = dirs + '_histo' +'_'+ targets[int(target)] +'.png'
            dirs = dirs + '_'+ targets[int(target)] +'.png'
        else:
            histogram_dirs = dirs + '_histo' +'_'+ targets[int(target)] +'.png'
            dirs = dirs + '_'+ targets[int(target)] +'.png'
    
    
        fig, ax = plt.subplots()

        # colors = ['g', 'r']
        colors = ['tab:green', 'tab:orange', 'tab:red'] # Added
        
        
        ### Added histogram to visualise the distribution of the weights
        # Determining the number of bins
        number_of_bins = 10 # np.around(np.sqrt(len(weights))) usually gives about 8 or 9
        print('The number of bins chosen is:',number_of_bins)
        # Plotting the graph
        plt.hist(weights, bins=number_of_bins)
        plt.xlabel('Learned attention weight')
        plt.ylabel('Number of images')
        plt.show()
        plt.savefig(histogram_dirs)


        for i, c, label in zip(label_ids, colors, label_names):
            # ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label)

            # ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label, marker=".", s=30) # Added
            ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label, marker="+", s=50) # Added
            
            ax.set_title('Attention weights')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        #for i, c, label in zip(target_ids, colors, target_names):
         #   ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label)
        
        
        ### To add or remove annotations, comment out the 2 lines below
        # for i, txt in enumerate(weights):
        #      ax.annotate(txt, (tsne_2d[i,0], tsne_2d[i,1]))
            
        plt.legend()
        plt.savefig(dirs)
        plt.cla()
        plt.clf()
        tsne=[]
        kmeans=[]




### End of "patient_specific_clustering.py" ####


#### The below is "show_intra_emnd.py" ####


if __name__ == "__main__":

    args = parse_args()

    print ('Called with args:')
    print (args)

    input_dim = (224,224,1)


    test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify

    checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'
    model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone

    
    model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load
    
    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)
  
    model_outputs = Model(inputs=model.input,
                                     outputs=[model.get_layer('multiply_1').output, model.get_layer('alpha').output])
    
    target_names= ['g', 'y', 'r']
    colors = ['g', 'y', 'r']
    target_ids = range(len(target_names))
    S = 4000
    s= 225
    for ibatch, batch in enumerate(test_set):
        
        print(batch[2][0])
        ins_emb, weights = model_outputs.predict_on_batch(x=batch[0])
        weights= np.squeeze(np.around(weights, decimals=4),1)
        
        
        # Adaptive Thresholding
        Top_10 = np.percentile(weights, 90)
        Bottom_10 = np.percentile(weights, 10)

        labels  = np.asarray([2 if i >= Top_10 else 1 if Top_10>i>= Bottom_10 else 0 for i in weights], dtype= 'int64') # Added
                
#         labels  = np.asarray([1 if i >= 0.01 else 0 for i in weights], dtype= 'int64')
    
        #kmeans = KMeans(2)
        #labels = kmeans.fit_predict(ins_emb)

        tsne = TSNE(n_components=2, perplexity=20)
        tsne_2d = tsne.fit_transform(ins_emb)
        #pca = PCA(2)  # project from 64 to 2 dimensions
        #projected = pca.fit_transform(ins_emb)
        
        # dirs = os.path.join(args.test_bags_samples, str(ibatch)) # Modify
        dirs = os.path.join(args["test_visualisation"], str(ibatch))

        targets= ['Non-COVID-19', 'COVID-19']
        
#         label_names= ['attention weights < 1%', 'attention weights >= 1%']
        label_names= ['Bottom 10%', 'Middle 80%', 'Top 10%'] # Added



        label_ids = range(len(label_names))
        target = np.mean(batch[1], axis=0, keepdims=False)
    
        if int(target)== 1:
            dirc= dirs + '_'+ targets[int(target)] +'.png'
            dirs= dirs + '_'+ targets[int(target)] +'_emb.png'
        else:
            dirc= dirs + '_'+ targets[int(target)] +'.png'
            dirs= dirs + '_'+ targets[int(target)] +'_emb.png'

        embed_map = np.zeros((S,S), 'float32')

        x= tsne_2d - np.min(tsne_2d)
        x= x/ np.max(x)

        for n, image in enumerate(batch[0]):

            #location
            a= np.ceil(x[n,0] * (S-s) +1)
            b= np.ceil(x[n,1] * (S-s) +1)
            a= int(a- np.mod(a-1,s) +1)
            b= int(b- np.mod(b-1,s) +1)

            if embed_map[a,b] != 0:
                continue
            I = np.squeeze(image, axis=2)
            I= ndimage.rotate(I,270, reshape=False)
            embed_map[a:a+s-1, b:b+s-1] = I;
        embed_map= ndimage.rotate(embed_map,90, reshape=False)
        fig = plt.figure()   
        plt.gray()
        plt.imshow(embed_map)
        plt.savefig(dirs)
        #plt.cla()
        #plt.clf()

        fig, ax = plt.subplots()     
        # colors = ['g', 'r']
        colors = ['tab:green', 'tab:orange', 'tab:red'] # Added


        for i, c, label in zip(label_ids, colors, label_names):
            # ax.scatter(x[labels == i, 0], x[labels == i, 1], c=c, label=label)
            ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label, marker="+", s=50) # Added

            ax.set_title('Attention weights') # CHECK
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        #for i, c, label in zip(target_ids, colors, target_names):
         #   ax.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label)
#         for i, txt in enumerate(weights):
#              ax.annotate(txt, (x[i,0], x[i,1]))
        plt.legend()
        plt.savefig(dirc)
        #plt.cla()
        #plt.clf()
        plt.show()
        import pdb
        pdb.set_trace
        tsne=[]
         #load embedding
        #close all;
        # load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne



#### End of "show_intra_emnd.py" ####


#### Start of "tsne.py" ####

if __name__ == "__main__":

    args = parse_args()

    print ('Called with args:')
    print (args)

    input_dim = (224,224,1)

    test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify

    checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'
    model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone

    
    model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load
    
    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)

    model_outputs = Model(inputs=model.input,
                                     outputs=model.get_layer('multiply_1').output)
  
    dirs = os.path.join(args["test_visualisation"],'tsne') # Added

    embed= []
    labels= []
    subject_id =[]
    for ibatch, batch in enumerate(test_set):
        
        ins_emb = model_outputs.predict_on_batch(x=batch[0])
        labels.append(np.mean(batch[1], axis=0, keepdims=False))
        embed.append(np.sum(ins_emb, axis=0, keepdims=True))
        subject_id.append(ibatch)
        
    labels = np.asarray(labels, dtype='int64')
    subject_id = np.asarray(subject_id, dtype='int64')

    tsne = TSNE(n_components=2, perplexity=25)
    
    tsne_2d = tsne.fit_transform(np.concatenate(embed, axis=0)) # to modify - throwing an error


    target_names= ['Non-COVID-19', 'COVID-19']
    target_ids = range(len(target_names))

    fig, ax = plt.subplots()
    # colors = 'g', 'r'
#     colors = ['tab:green', 'tab:orange', 'tab:red'] # Added
    colors = ['tab:green', 'tab:red']
    
    for i, c, label in zip(target_ids, colors, target_names):
#         plt.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label)

        plt.scatter(tsne_2d[labels == i, 0], tsne_2d[labels == i, 1], c=c, label=label, marker="+", s=50) # Added - TRY THIS instead of line above?

#     for i, txt in enumerate(subject_id): # Remove to remove annotation
#         ax.annotate(txt, (tsne_2d[i,0], tsne_2d[i,1]))

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()
    plt.savefig(dirs) # added



#### End of "tsne.py" ####



#### Start of "vis_attention.py" ####

import pdb
from sklearn.model_selection import KFold


if __name__ == "__main__":

    args = parse_args()

    print ('Called with args:')
    print (args)

    input_dim = (224,224,1)


    test_set = np.load('/app/Alex/model_saving_folder/Covid_Net_ResNet18/data/fold_1_test_bags.npy', allow_pickle=True) # To modify

    checkpoint_path='/app/Alex/model_saving_folder/Covid_Net_ResNet18/Saved_model/hd5_files/_Batch_size_1fold_1.ckpt'
    model = covid_ResNet18(input_dim, args, useMulGpu=False)  # model c) with ResNet18 backbone

    
    model.load_weights(checkpoint_path) # change it to checkpoint path as per: https://www.tensorflow.org/tutorials/keras/save_and_load
    
    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC = test_eval(model, test_set) # Gives test loss and accuracy

    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)

 # pdb.set_trace()

    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer('model_1').get_layer('alpha').output) # Modify
    
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('alpha').output)
  

    labels= ['Non-COVID', 'COVID']
    for ibatch, batch in enumerate(test_set):
        intermediate_output = intermediate_layer_model.predict(batch[0])

        # dirs = os.path.join(args.test_bags_samples, str(ibatch)) # Modify
        dirs = os.path.join(args["test_visualisation"], str(ibatch))

        if not os.path.exists(dirs):
            os.mkdir(dirs)
            
            
        for im in range(batch[0].shape[0]):
            fig, ax = plt.subplots()
            ax.set_axis_off()
            img = np.squeeze(batch[0][im], 2)
            ax.imshow(img, cmap='gray', interpolation='bilinear')
            ax.text(70, 12, labels[batch[1][im]]+':%.6f' %np.around(intermediate_output[im], 6), color='r', fontsize=15)
            extent1 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(os.path.join(dirs, batch[2][im]), bbox_inches=extent1)
            plt.cla()
            plt.clf()



### End of "vis_attention.py" ####



print('end of file')