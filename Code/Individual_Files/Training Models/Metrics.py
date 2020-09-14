# Metrics used to evaluate the models

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

