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

