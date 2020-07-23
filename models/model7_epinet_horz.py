#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:36:18 2020

@author: rgupta
"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" import keras libraries"""
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D ,Cropping2D
from keras.layers import concatenate,Add,Reshape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger,EarlyStopping
from keras.models import load_model, Model, Sequential
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop,Adam

""" import image handle library """

import numpy as np
import cv2
import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io
from skimage.transform import resize
import h5py
import natsort
from numpy import asarray
from numpy import save

"""argument to run the whole program from console"""


lens_type = sys.argv[1]
print(lens_type)

"""argument to run the whole program from console"""
EPOCH = 70
BATCH_SIZE = 16
adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=False)

""" path where data for training and prediction are saved """
data_set_path = "" 

""" path where our model weights wull be saved after training """

checkpoint_path = ''

""" here we are joining folders inside train and prediction folder to the path for data retrieval"""

if lens_type == 'seven':
    
    training_set_path = os.path.join(data_set_path, 'train')
    prediction_set_path = os.path.join(data_set_path,'prediction')
    
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_horz_path = os.path.join(training_set_path,'horizontal')
    
    pred_horz_path = os.path.join(prediction_set_path,'horizontal')
    
if lens_type == 'three':
    
    training_set_path = os.path.join(data_set_path, 'train')
    prediction_set_path = os.path.join(data_set_path,'prediction')
    

    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_horz_left_path = os.path.join(training_set_path,'left')
    train_horz_right_path = os.path.join(training_set_path,'right')
    
    
    pred_horz_left_path = os.path.join(prediction_set_path,'left')
    pred_horz_right_path = os.path.join(prediction_set_path,'right')

    

"""Load image from folder"""
def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list = natsort.natsorted(set_list)
#    print(set_list)
    for set_path in set_list:
        img = cv2.imread(os.path.join(folder,set_path),cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        red = img[:,:,2]
        all_images.append(red)
    return np.array(all_images)

"""functions for fully convolutional neural network"""


def layer1_multistream(input_dim1,input_dim2,input_dim3):    
    seq = Sequential()
    ''' Multi-Stream layer : Conv - Relu - Conv - BN - Relu  '''

#    seq.add(Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
    for i in range(3):
        seq.add(Conv2D(70,(2,2),input_shape=(input_dim1, input_dim2,input_dim3), padding='valid', name='S1_c1%d' %(i) ))
        seq.add(Activation('relu', name='S1_relu1%d' %(i))) 
        seq.add(Conv2D(70,(2,2), padding='valid', name='S1_c2%d' %(i) )) 
        seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
        seq.add(Activation('relu', name='S1_relu2%d' %(i))) 

    seq.add(Reshape((input_dim1-6,input_dim2-6,70)))

    return seq  
def layer2_merged(input_dim1,input_dim2,input_dim3):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    
    seq = Sequential()
    
    for i in range(7):
        print(i)
        seq.add(Conv2D(280,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
        seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
        seq.add(Conv2D(280,(2,2), padding='valid', name='S2_c2%d' % (i))) 
        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        seq.add(Activation('relu', name='S2_relu2%d' %(i)))
        i+=1
          
    return seq  
   

def layer3_last(input_dim1,input_dim2,input_dim3):   
    ''' last layer : Conv - Relu - Conv ''' 
    
    seq = Sequential()
    
    for i in range(1):
        seq.add(Conv2D(280,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(Activation('relu', name='S3_relu1%d' %(i)))
        
    seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last')) 

    return seq 

""" function for root mean squared error """
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


 """ making our fully convolutional neural network model for seven-lens Input data"""  

if lens_type == 'seven':
    
    def define_epinet(sz_input,sz_input2,view_n,conv_depth):
    
        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
    #    shape=(None, input_shape[1], input_shape[2], input_shape[3])
        horz = Input(shape=(sz_input,sz_input2,len(view_n)))
        
      
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        middle_1=layer1_multistream(sz_input,sz_input2,len(view_n))(horz)
        print(middle_1)
        tensor_shape= middle_1.get_shape()
        batch_size=tensor_shape[3]
        print(batch_size)
      
       
        
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input,sz_input2,batch_size)(middle_1)
        print(mid_merged_)
    
        
        ''' Last Conv layer : Conv - Relu - Conv '''
        output=layer3_last(sz_input-18,sz_input2-18,280)(mid_merged_)
        print(output)
    
        
        
        up_1 = Conv2D(filters=62,kernel_size=2,padding='same')(output)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((4,0),(9,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
    
    #    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(up_1_batch)
    #    decoded_cropping = Cropping2D((2,2))(decoded)
    
        up_2 = Conv2D(filters=7,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
    
        
        opt = RMSprop(lr=0.1**4)
        model = Model(inputs = [horz], outputs = [up_2])
        model.compile(optimizer=opt, loss='mse',metrics=[rmse])
    #    model_512.summary() 
        
        return model

    # Load train and test set from folder path
    print('Load train_horz')
    train_horz = load_images_from_folder(train_horz_path)
    #train_horz = np.expand_dims(train_horz, axis=3)
    
    print('Load train_disp')
    train_disp = load_images_from_folder(train_disp_path)
    #train_disp = np.expand_dims(train_disp, axis=3)
    
   
    
    print('load pred_horz')
    pred_horz=load_images_from_folder(pred_horz_path)
    #plt.imshow(pred_horz[10])
    #pred_horz = np.expand_dims(pred_horz,axis = 3)
    
    # print shapes
    print(train_horz.shape)
    
    print(train_disp.shape)
    print(pred_horz.shape)
 
""" initializong list and varaible for transforming all the sevem Images into a single Image of seven channel for 
training and for ground truth data we are just converting single ground truth lens into seven channel of single Image
(all the other six channek are empty just we need main ground truth lens for output label) bcoz of the network nature we have to give same number of channel as input and output labels
 """    
    
    
    w = []
    w1= []
    j = 0
    z = 0
    
    for i in range(0,len(train_horz)):
        if i<(7+j*7):
            w.append(train_horz[i])    
            z+=1
            if z % 7 ==0:
                image= np.zeros((40, 35, 7),np.float32)
                print(image.shape)
                image[:,:,0] = w[0]
                image[:,:,1] = w[1]
                image[:,:,2] = w[2]
                image[:,:,3] = w[3]
                image[:,:,4] = w[4]
                image[:,:,5] = w[5]
                image[:,:,6] = w[6]
                print(image.shape)
                w1.append(image)
                w=[]
                j+=1
                z=0
    
    w1= np.array(w1)
    w1.shape
    len(w1)
    w2=[]
    for i in range(0,len(train_disp)):
        image=np.zeros((40,35,7),np.float32)
        image[:,:,0]=train_disp[i]
        w2.append(image)
        
    len(w2)
    w2=np.array(w2)
    print(w2.shape)
               
    print(train_horz[0]) 
    
    # Number of channels 

    Setting02_AngualrViews = np.array([0,1,2,3,4,5,6])

 """ making our fully convolutional neural network model for three-lens Input data"""  

if lens_type == 'three':
    def define_epinet(sz_input,sz_input2,view_n,conv_depth):
        
        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
    #    shape=(None, input_shape[1], input_shape[2], input_shape[3])
        horz_left = Input(shape=(sz_input,sz_input2,len(view_n)))
        horz_right = Input(shape=(sz_input,sz_input2,len(view_n)))
        
    
    
        
      
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        middle_1=layer1_multistream(sz_input,sz_input2,len(view_n))(horz_left)
        middle_2=layer1_multistream(sz_input,sz_input2,len(view_n))(horz_right)
        
        merge = concatenate([middle_1,middle_2])
        print(1)
        print(merge)
    
        
        tensor_shape= merge.get_shape()
        batch_size=tensor_shape[3]
        print(batch_size)
      
       
        
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input,sz_input2,batch_size)(merge)
        print(mid_merged_)
    

        
        ''' Last Conv layer : Conv - Relu - Conv '''
        output=layer3_last(sz_input-18,sz_input2-18,280)(mid_merged_)
        print(output)
    
        
        
        up_1 = Conv2D(filters=62,kernel_size=2,padding='same')(output)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((4,0),(9,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
    

    
        up_2 = Conv2D(filters=3,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
    
        
        opt= RMSprop(lr=0.1**4)
        model = Model(inputs = [horz_left,horz_right], outputs = [up_2])
        model.compile(optimizer=opt, loss='mse',metrics=[rmse])
    #    model_512.summary() 
        
        return model

 """ Load train and prediction set from folder path """


    print('Load train_horz_left')
    train_horz_left = load_images_from_folder(train_horz_left_path)
    #train_horz = np.expand_dims(train_horz, axis=3)
    
    print('Load train_horz_right')
    train_horz_right = load_images_from_folder(train_horz_right_path)
    
    print('Load train_disp')
    train_disp = load_images_from_folder(train_disp_path)
    #train_disp = np.expand_dims(train_disp, axis=3)
    

    
    print('load pred_left_horz')
    pred_horz_left=load_images_from_folder(pred_horz_left_path)
    #plt.imshow(pred_horz[10])
    #pred_horz = np.expand_dims(pred_horz,axis = 3)
    
    print('load pred_right_horz')
    pred_horz_right=load_images_from_folder(pred_horz_right_path)
    
    ## print_shapes
    print(train_horz_left.shape)
    print(train_horz_right.shape)
    print(train_disp.shape)
    

    
    print(pred_horz_left.shape)
    print(pred_horz_right.shape)
    
    
     """ initializong list and varaible for transforming all the three Images into a single Image of three channel for training
 and for ground truth we dont need three images just main ground length is required as output label but we need to transform 
 it into three channel image out of which two channel are empty same as we did with seven lens data """
    ### tranforming train_left
    train_left = []
    train_left1= []
    j = 0
    z = 0
    
    for i in range(0,len(train_horz_left)):
        if i<(3+j*3):
            train_left.append(train_horz_left[i])    
            z+=1
            if z % 3 ==0:
                image= np.zeros((40, 35, 3),np.float64)
    #            print(image.shape)
                image[:,:,0] = train_left[0]
                image[:,:,1] = train_left[1]
                image[:,:,2] = train_left[2]
    #            print(image.shape)
                train_left1.append(image)
                train_left=[]
                j+=1
                z=0
    
    train_left1= np.array(train_left1)
    train_left1.shape
    len(train_left1)
    
    ### transforming train_right
    
    train_right = []
    train_right1= []
    j = 0
    z = 0
    
    for i in range(0,len(train_horz_right)):
        if i<(3+j*3):
            train_right.append(train_horz_right[i])    
            z+=1
            if z % 3 ==0:
                image= np.zeros((40, 35, 3),np.float64)
    #            print(image.shape)
                image[:,:,0] = train_right[0]
                image[:,:,1] = train_right[1]
                image[:,:,2] = train_right[2]
    #            print(image.shape)
                train_right1.append(image)
                train_right=[]
                j+=1
                z=0
    
    train_right1= np.array(train_right1)
    train_right1.shape
    len(train_right1)        
    
    ## transforming disparity data
    
    disparity_train=[]
    for i in range(0,len(train_disp)):
        image=np.zeros((40,35,3),np.float64)
        image[:,:,0]=train_disp[i]
        disparity_train.append(image)
        
    len(disparity_train)
    disparity_train=np.array(disparity_train)
    print(disparity_train.shape)
    
    # Number of channels 

    Setting02_AngualrViews = np.array([0,1,2])
    
"""number of convolutional depths"""
conv_depth=7

""" build model """
model = define_epinet(40,35,
                      Setting02_AngualrViews,
                      conv_depth)

"""set early stopping criteria"""
pat=5#this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss',patience=pat, verbose=1)
""" here we are saving all the model weights and everything about the model so that we can load out trained model again"""

## Add Learning option and learning
checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = True)
""" passing path where we want to save our logger , try to save everything in same path pass same path as
checkpoint path passed before and save the logger with name log and extension .csv"""
logger = CSVLogger(filename='/home/rgupta/Desktop/results_sevend/epinet/log.csv')
""" to see the layers,number of parameters and details of the model """                

model.summary()

""" fitting the model for training """

if lens_type == 'seven':
    history = model.fit(w1,
                        w2,
                        epochs = EPOCH,
                        validation_split=0.2,
                        batch_size = BATCH_SIZE,
                        callbacks=[early_stopping,logger,checkpoint])
    
if lens_type == 'three':
    history = model.fit([train_left1,train_right1],
                    disparity_train,
                    epochs = EPOCH,
                    validation_split=0.2,
                    batch_size = BATCH_SIZE,
                    callbacks=[early_stopping,logger,checkpoint])
    
""" graphs to check the training and validation progress of model """

# draw and save result

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss','val_loss'],loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
"""path for saving the graph with name train_loss and .png as extension 
we have used same folder to save everything which we need to save """
plt.savefig('/home/rgupta/Desktop/results_sevend/epinet/train_loss.png')
plt.close()

## graph for rmse###

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.grid()
plt.title('model performance')
plt.ylabel('metrics')
plt.xlabel('epoch')

""" saving the graph with train_acc name with .png as extension """

plt.savefig('/home/rgupta/Desktop/results_sevend/epinet/train_acc.png')
plt.close()


""" if we want to load the weights of our trained model and we can use it for prediction later on different Image 
even we can also trained our already trained model. specify path where our checkpoint.h5 file is saved, just
cahnge the path and the name of our checkpoint file is checkpoint.h5 so we do not need to change it """
#model.load_weights('/home/rgupta/Desktop/results/checkpoint.h5')

""" we also have to transform prediction data as we did for training data but here we don not need ground truth data 
as for prediction we dont give output label """

## transforming prediction data##
if lens_type == 'seven':
    
    pred = []
    pred1= []
    j2 = 0
    z2 = 0
    
    for i in range(0,len(pred_horz)):
        if i<(7+j2*7):
            pred.append(pred_horz[i])    
            z+=1
            if z % 7 ==0:
                image= np.zeros((40, 35, 7),np.float32)
                print(image.shape)
                image[:,:,0] = pred[0]
                image[:,:,1] = pred[1]
                image[:,:,2] = pred[2]
                image[:,:,3] = pred[3]
                image[:,:,4] = pred[4]
                image[:,:,5] = pred[5]
                image[:,:,6] = pred[6]
                print(image.shape)
                pred1.append(image)
                pred=[]
                j2+=1
                z2=0
    
    pred1= np.array(pred1)
    pred1.shape
    len(pred1)
    
if lens_type == 'three':
    ## transforming prediction_left data##
    pred_left = []
    pred_left1= []
    j2 = 0
    z2 = 0
    
    for i in range(0,len(pred_horz_left)):
        if i<(3+j2*3):
            pred_left.append(pred_horz_left[i])    
            z+=1
            if z % 3 ==0:
                image= np.zeros((40, 35, 3),np.float64)
                print(image.shape)
                image[:,:,0] = pred_left[0]
                image[:,:,1] = pred_left[1]
                image[:,:,2] = pred_left[2]
                print(image.shape)
                pred_left1.append(image)
                pred_left=[]
                j2+=1
                z2=0
    
    pred_left1= np.array(pred_left1)
    pred_left1.shape
    len(pred_left1)
    
    
    ## transforming prediction_right data##
    pred_right = []
    pred_right1= []
    j2 = 0
    z2 = 0
    
    for i in range(0,len(pred_horz_right)):
        if i<(3+j2*3):
            pred_right.append(pred_horz_right[i])    
            z+=1
            if z % 3 ==0:
                image= np.zeros((40, 35, 3),np.float64)
                print(image.shape)
                image[:,:,0] = pred_right[0]
                image[:,:,1] = pred_right[1]
                image[:,:,2] = pred_right[2]
                print(image.shape)
                pred_right1.append(image)
                pred_right=[]
                j2+=1
                z2=0
    
    pred_right1= np.array(pred_right1)
    pred_right1.shape
    len(pred_right1)
    




#####  making predictions ####################

""" Initialize list for saving predictions """

disparity_list = []
if lens_type == 'seven':
    new_predictions = model.predict(pred1)
    
if lens_type == 'three':
    new_predictions = model.predict([pred_left1,pred_right1])
    


for i in range(len(new_predictions)):
    disparity_list.append(new_predictions[i])
len(disparity_list)


"""saving the prediction result in a list using numpy """

disparity_list = asarray(np.array(disparity_list))
len(disparity_list)


"""path for saving the numpy arrays with name disparity_list and .npy as extension and here we need to specify the numpy
array name (in our case name is disparity_list only) that we want to save """

save('/home/rgupta/Desktop/results_sevend/epinet/disparity_list.npy',disparity_list)