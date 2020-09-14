
import time
start = time.time()

# Select a single GPU in Keras
# Scenario: You have multiple GPUs on a single machine running Linux, but you want to use just one. By default, Keras allocates memory to all GPUs unless you specify otherwise. You use a Jupyter Notebook to run Keras with the Tensorflow backend.

# Here’s how to use a single GPU in Keras with TensorFlow
# Run this bit of code in a cell right at the start of your notebook (before importing tensorflow or keras).
 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use: "0", "1" or "2" or "3";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


## Hack to make other Keras backbones work
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
                    use_bias=True, use_gated=True, **kwargs):
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




destination_folder= '/app/Alex/Images_Destination_Folder'  # Adapt to cluster




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
from keras import backend as K


# Precision

def precision_m(y_true, y_pred): # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Sensitivity/Recall (measures the proportion of actual positives that are correctly identified as such (tp / (tp + fn))) https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.metrics.sensitivity_score.html


def recall_m(y_true, y_pred): # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



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

    args = dict(init_lr = 1e-1, 
    weight_decay = 0.0005,
    momentum = 0.9,
    max_epoch=50, # Keep modify it depending on the model and how quickly it overfits
    useGated = True,             
                             
    train_bags_samples = model_saving_folder + '/save_train_bags',
    test_bags_samples = model_saving_folder + '/test_results',
    model_id = model_saving_folder +'/Simple_ConvNet') # Need to modify this since we changed dataset
 
    if not os.path.exists(args["model_id"]):
        os.mkdir(args["model_id"])
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
    test_precision_manually = np.zeros((num_test_batch, 1), dtype=float)
    test_recall_manually = np.zeros((num_test_batch, 1), dtype=float)


    for ibatch, batch in enumerate(test_set): # Throwing tracing error      
        
#         print('x = batch.shape = ',batch.shape) # To help debug
#         print('x = batch[0].shape = ',batch[0].shape) # To help debug
#         print('y = batch[1].shape = ',batch[1].shape) # To help debug
#         print('y = batch[1] = ',batch[1]) # To help debug

        result = model.test_on_batch(x=batch[0], y=batch[1][:1]) # y=batch[1] is a vector of all ones or all zeros for the bag. Take first element to have the right dimension to compare to y_true
#         result = model.evaluate(x=batch[0], y=batch[1]) # y=batch[1] is a vector of all ones or all zeros for the bag. Take first element to have the right dimension to compare to y_true

#         result = model.test_on_batch(x=batch[0], y=np.mean(batch[1],keepdims=True))
        
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]

#         print('In test_eval, len(result) = ',len(result))
        test_precision[ibatch] = result[2]
        test_recall[ibatch] = result[3]
        test_specificity[ibatch] = result[4]
        test_AUC[ibatch] = result[5]
        
        test_precision_manually[ibatch] = result[6]
        test_recall_manually[ibatch] = result[7]

#     y_preds = np.concatenate([model.predict(batch[0]).ravel() for batch in test_set]) # added
#     y_trues = np.concatenate([batch[1] for batch in test_set]) # added
#     plot_roc_curve(y_trues, y_preds) # added

    return np.mean(test_loss), np.mean(test_acc), np.mean(test_precision), np.mean(test_recall), np.mean(test_specificity), np.mean(test_AUC), np.mean(test_precision_manually), np.mean(test_recall_manually)





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

    
    ### Path for tf.keras: https://www.tensorflow.org/tutorials/keras/save_and_load
#     checkpoint_path = "training_1/cp.ckpt" 

    checkpoint_path = args["model_id"]+"/Saved_model/" + "hd5_files/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + ".ckpt"
    
    checkpoint_dir = os.path.dirname(checkpoint_path)

    
    model_name = args["model_id"]+"/Saved_model/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + "best.hd5"



#     checkpoint_fixed_name = ModelCheckpoint(model_name,
#                                             monitor='val_loss', verbose=1, save_best_only=True,
#                                             save_weights_only=True, mode='auto', save_freq='epoch')


    checkpoint_fixed_name = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')


#     # Create a callback that saves the model's weights
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)    

    EarlyStop = EarlyStopping(monitor='val_loss', patience=2)

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

#     model = simple_conv_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # Simple ConvNet baseline
    
#     model = instance_based_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model a)
#     model = embedding_based_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model b)
    model = covid_net(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c)

#     model = covid_ResNet18(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with ResNet18 backbone
    
    
#     model = covid_ResNet50(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with ResNet50 backbone
    

#     model = covid_InceptionV3(input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with InceptionV3 backbone
    

#     model = covid_SqueezeExcite_ResNet50(input_dim, args, normalised_inv_freq, bottleneck=True, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze-Excite ResNet50 backbone instead of the standard CNN backbone

#     model = covid_SqueezeExcite_InceptionV3(input_dim, args, normalised_inv_freq, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze-Excite InceptionV3 backbone instead of the standard CNN backbone


    ## Test out the below:
#     model = covid_SEInceptionResNetV2(input_dim, args, normalised_inv_freq, useMulGpu=False) # Model c) in Ilse et. al (2018) with Squeeze Excite InceptionResNetV2 backbone instead of the standard CNN backbone
    

    ## Pick DenseNet version
    blocks = [6, 12, 24, 16] # DenseNet121 Model
    # blocks = [6, 12, 32, 32] # DenseNet169 Model
    # blocks = [6, 12, 48, 32] # DenseNet201 Model
    # blocks = [TO BE SPECIFIED BY USER] # DenseNet Model
    
#     model = covid_DenseNet(blocks, input_dim, args, normalised_inv_freq, useMulGpu=False)  # model c) with DenseNet backbone

    
    ### End of model selection panel ### 


    # train model
    t1 = time.time()
    # for epoch in range(args.max_epoch):

    model_name = train_eval(model, train_set, irun, ifold)

    #print("load saved model weights")
    #model.load_weights(model_name)


    test_loss, test_acc, test_precision, test_recall, test_specificity, test_AUC, test_precision_manually, test_recall_manually = test_eval(model, test_set) # Gives test loss and accuracy and other metrics
        
    print('test_loss:%.3f' %test_loss)
    print('test_acc:%.3f' %test_acc)
    print('test_precision:%.3f' %test_precision)
    print('test_recall:%.3f' %test_recall)
    print('test_specificity:%.3f' %test_specificity)
    print('test_AUC:%.3f' %test_AUC)
    
    
    print('test_precision_manually_calculated:%.3f' %test_precision_manually)
    print('test_recall_manually_calculated:%.3f' %test_recall_manually)

    #t2 = time.time()
    #out_test = open('Results/' + 'test_results.txt', 'w')
    #out_test.write("fold{} run_time:{:.3f} min  test_acc:{:.3f} ".format(ifold, (t2 - t1) / 60.0, test_acc,))
    #out_test.write("\n")

    return model_name

print('Model is now ready to train!')

print('Training the model...')

if __name__ == "__main__":

    args = parse_args() 

    print ('Called with args:') # Not needed in Colab notebook
    print (args) # Not needed in Colab notebook

    input_dim = (224,224,1) # [TBU] Change depending on dataset?

    run = 1
    n_folds = 5
    acc = np.zeros((run, n_folds), dtype=float)
    # data_path = './data/data/' # [#TBU]
    data_path = destination_folder

    for irun in range(run):
        dataset = load_dataset(dataset_path=data_path, n_folds=n_folds, rand_state=irun)
#         for ifold in range(n_folds): # need to get rid of this For Loop. Do only one fold
        ifold=0 # Pick fold number here
        print('run=', irun, '  fold=', ifold)
        #acc[irun][ifold] = model_training(input_dim, dataset[ifold], irun, ifold)
        _ = model_training(input_dim, dataset[ifold], irun, ifold)
        # print ('mi-net mean accuracy = ', np.mean(acc))
        # print ('std = ', np.std(acc))
        
        end = time.time()
        print(end - start)