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
        # load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)