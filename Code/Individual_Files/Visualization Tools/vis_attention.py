import pdb
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


