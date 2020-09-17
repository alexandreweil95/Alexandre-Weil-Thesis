# Main file for training the models

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
    max_epoch=50,
    useGated = True,             
                             
    train_bags_samples = model_saving_folder + '/save_train_bags',
    test_bags_samples = model_saving_folder + '/test_results',
    model_id = model_saving_folder +'/Simple_ConvNet') # Need to modify this for every model, so that the results of different models get saved to different folders
 
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
                curr_label = np.ones(len(img), dtype=np.uint8)
                num_pos += 1
            else:
                curr_label = np.zeros(len(img), dtype=np.uint8)
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

    
    checkpoint_path = args["model_id"]+"/Saved_model/" + "hd5_files/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + ".ckpt"
    
    checkpoint_dir = os.path.dirname(checkpoint_path)

    
    model_name = args["model_id"]+"/Saved_model/" + "_Batch_size_" + str(batch_size) + "fold_" + str(ifold) + "best.hd5"

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
#     np.save(os.path.join(dirc, 'fold_{}_train_bags.npy'.format(ifold)), train_set)
    np.save(os.path.join(dirc, 'fold_{}_test_bags.npy'.format(ifold)), test_set) 
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

    print ('Called with args:')
    print (args) 

    input_dim = (224,224,1)

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
