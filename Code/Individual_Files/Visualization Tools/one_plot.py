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

