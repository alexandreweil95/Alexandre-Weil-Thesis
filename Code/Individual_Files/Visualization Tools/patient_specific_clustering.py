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


