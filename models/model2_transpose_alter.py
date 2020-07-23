#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:41:02 2020

@author: rgupta
"""





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Importing keras libraries"""
import keras
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D 
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger, EarlyStopping
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

"""Importing Image handling libraries"""
import numpy as np
import natsort
import math
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
from numpy import load,save
from numpy import asarray
from numpy import save

"""argument to run the whole program from console"""

lens_type = sys.argv[1]
print(lens_type)

"""setup for learning"""
EPOCH = 50
BATCH_SIZE = 64
adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=False)


# setup data path for train and prediction 
""" path where data for training and prediction are saved """
data_set_path = "" 

""" path where our model weights wull be saved after training """

checkpoint_path = ''


""" here we are joining folders inside train and prediction folder to the path for data retrieval"""
if lens_type == 'four':
    
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(data_set_path,'dis_images1')
    train_middle_path = os.path.join(data_set_path,'middle')
    train_right_path = os.path.join(data_set_path,'right')
    train_left_path = os.path.join(data_set_path,'left')
    train_top_path = os.path.join(data_set_path,'top')
    train_bottom_path = os.path.join(data_set_path,'bottom')
    
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_middle_path = os.path.join(prediction_set_path,'middle')
    pred_right_path = os.path.join(prediction_set_path,'right')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_top_path = os.path.join(prediction_set_path,'top')
    pred_bottom_path = os.path.join(prediction_set_path,'bottom')


# Load image from folder
def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list=natsort.natsorted(set_list)
    for set_path in set_list:
        img = cv2.imread(os.path.join(folder,set_path),cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        red = img[:,:,2]
        all_images.append(red)
    return np.array(all_images)

""" making our fully convolutional neural network model for two-lens Input data"""  
if lens_type == 'two':
    def disparity_cnn_model(input_shape):
        shape=(None, input_shape[1], input_shape[2],input_shape[3])
        middle = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        top = Input(batch_shape=shape)
        bottom =  Input(batch_shape=shape)
        
        middle_1 = Conv2D(filters=32, kernel_size=3,padding='same')(middle)
        middle_1_pool = MaxPooling2D(2)(middle_1)
        middle_1_activate = Activation('relu')(middle_1_pool)
        
    
        middle_2 = Conv2D(filters=62, kernel_size=3,padding='same')(middle_1_activate)
        middle_2_pool = MaxPooling2D(2)(middle_2)
        middle_2_activate = Activation('relu')(middle_2_pool)
        
    
        middle_3 = Conv2D(filters=92, kernel_size=3,padding='same')(middle_2_activate)
        middle_3_activate = Activation('relu')(middle_3)
        
    
        right_1 = Conv2D(filters=32, kernel_size=3,padding='same')(right)
        right_1_pool = MaxPooling2D(2)(right_1)
        right_1_activate = Activation('relu')(right_1_pool)
        
    
        right_2 = Conv2D(filters=62, kernel_size=3,padding='same')(right_1_activate)
        right_2_pool = MaxPooling2D(2)(right_2)
        right_2_activate = Activation('relu')(right_2_pool)
        
    
        right_3 = Conv2D(filters=92, kernel_size=3,padding='same')(right_2_activate)
        right_3_activate = Activation('relu')(right_3)
        
        merge = concatenate([middle_3_activate,right_3_activate])
        
        left_1 = Conv2D(filters=32, kernel_size=3,padding='same')(left)
        left_1_pool = MaxPooling2D(2)(left_1)
        left_1_activate = Activation('relu')(left_1_pool)
        
    
        left_2 = Conv2D(filters=62, kernel_size=3,padding='same')(left_1_activate)
        left_2_pool = MaxPooling2D(2)(left_2)
        left_2_activate = Activation('relu')(left_2_pool)
        
    
        left_3 = Conv2D(filters=92, kernel_size=3,padding='same')(left_2_activate)
        left_3_activate = Activation('relu')(left_3)
        
        merge1 = concatenate([middle_3_activate,left_3_activate])
        
        top_1 = Conv2D(filters=32, kernel_size=3,padding='same')(top)
        top_1_pool = MaxPooling2D(2)(top_1)
        top_1_activate = Activation('relu')(top_1_pool)
        
    
        top_2 = Conv2D(filters=62, kernel_size=3,padding='same')(top_1_activate)
        top_2_pool = MaxPooling2D(2)(top_2)
        top_2_activate = Activation('relu')(top_2_pool)
        
    
        top_3 = Conv2D(filters=92, kernel_size=3,padding='same')(top_2_activate)
        top_3_activate = Activation('relu')(top_3)
        
        merge2 = concatenate([middle_3_activate,top_3_activate])
        
    
        
        
        bottom_1 = Conv2D(filters=32, kernel_size=3,padding='same')(bottom)
        bottom_1_pool = MaxPooling2D(2)(bottom_1)
        bottom_1_activate = Activation('relu')(bottom_1_pool)
        
    
        bottom_2 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_1_activate)
        bottom_2_pool = MaxPooling2D(2)(bottom_2)
        bottom_2_activate = Activation('relu')(bottom_2_pool)
        
    
        bottom_3 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_2_activate)
        bottom_3_activate = Activation('relu')(bottom_3)
        
        merge3 = concatenate([middle_3_activate,bottom_3_activate])
        
    
        
    
        final_merge = concatenate([merge,merge1,merge2,merge3])
    
        merge_1 = Conv2DTranspose(filters=62, kernel_size=3,strides = (2,2), padding='same')(final_merge)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(merge_1)
        merge_1_activate = Activation('relu')(zero_padd)
        
    
        merge_2 = Conv2DTranspose(filters=22, kernel_size=3,strides=(2,2), padding='same')(merge_1_activate)
        zero_padd1 = ZeroPadding2D(padding=((0,0),(1,0)))(merge_2)
        merge_2_activate = Activation('relu')(zero_padd1)
        
    
        merge_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1),padding='same')(merge_2_activate)
        merge_3_activate = Activation('relu')(merge_3)
    
        model = Model([middle,right,left,top,bottom], merge_3_activate)
    
        return model

    # Load train and prediction set from folder path
    print('Load train_middle')
    train_middle = load_images_from_folder(train_middle_path)
    train_middle = np.expand_dims(train_middle, axis=3)
    print('Load train_left')
    train_left = load_images_from_folder(train_left_path)
    train_left = np.expand_dims(train_left, axis=3)
    print('Load train_right')
    train_right = load_images_from_folder(train_right_path)
    train_right = np.expand_dims(train_right, axis=3)
    print('Load train_top')
    train_top = load_images_from_folder(train_top_path)
    train_top = np.expand_dims(train_top, axis=3)
    print('Load train_bottom')
    train_bottom = load_images_from_folder(train_bottom_path)
    train_bottom = np.expand_dims(train_bottom, axis=3)
    print('Load train_disp')
    train_disp = load_images_from_folder(train_disp_path)
    train_disp = np.expand_dims(train_disp, axis=3)
    
    print('load pred_middle')
    pred_middle=load_images_from_folder(pred_middle_path)
    #plt.imshow(pred_left[10])
    pred_middle = np.expand_dims(pred_middle,axis = 3)
    print('load pred_left')
    pred_left=load_images_from_folder(pred_left_path)
    #plt.imshow(pred_left[10])
    pred_left = np.expand_dims(pred_left,axis = 3)
    print('load pred_right')
    pred_right=load_images_from_folder(pred_right_path)
    #plt.imshow(pred_right[10])
    pred_right = np.expand_dims(pred_right,axis = 3)
    print('load pred_top')
    pred_top=load_images_from_folder(pred_top_path)
    #plt.imshow(pred_right[10])
    pred_top = np.expand_dims(pred_top,axis = 3)
    
    print('load pred_bottom')
    pred_bottom=load_images_from_folder(pred_bottom_path)
    pred_bottom = np.expand_dims(pred_bottom,axis = 3)
    

    print(train_middle.shape)
    print(train_left.shape)
    print(train_right.shape)
    print(train_top.shape)
    print(train_bottom.shape)
    print(train_disp.shape)
    
    print(pred_middle.shape)
    print(pred_top.shape)
    print(pred_bottom.shape)
    print(pred_left.shape)
    print(pred_right.shape)

# build model
model = disparity_cnn_model(train_left.shape)


""" function for defining root mean sqyared error"""
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

"""compiling the model"""
model.compile(loss='mse',
              optimizer=adam, metrics = [rmse])
"""set early stopping criteria"""
pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss',patience=pat, verbose=1)
## Add Learning option and learning
""" here we are saving all the model weights and everything about the model so that we can load out trained model again"""

checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = True)
""" passing path where we want to save our logger , try to save everything in same path pass same path as
checkpoint path passed before and save the logger with name log and extension .csv"""
logger = CSVLogger(filename='#path/log.csv')

""" to see the layers,number of parameters and details of the model """                
model.summary()

"""fitting the model for training"""
history = model.fit([train_middle,train_left, train_right,train_top,train_bottom],
                    train_disp,
                    epochs = EPOCH,
                    validation_split=0.2,
                    batch_size = BATCH_SIZE,
                    callbacks=[checkpoint,logger,early_stopping])

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

plt.savefig('/home/rgupta/Desktop/six_lens_result/result3/train_loss.png')
plt.close()

## draw and save rmse graph


plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['train_rmse','val_rmse'],loc = 'best')
plt.grid()
plt.title('model performance')
plt.ylabel('metrics')
plt.xlabel('epoch')
""" saving the graph with train_acc name with .png as extension """
plt.savefig('/home/rgupta/Desktop/six_lens_result/result3/train_acc.png')
plt.close()


""" if we want to load the weights of our trained model and we can use it for prediction later on different Image 
even we can also trained our already trained model. specify path where our checkpoint.h5 file is saved, just
cahnge the path and the name of our checkpoint file is checkpoint.h5 so we do not need to change it """


#model1 = model.load_weights('/home/rgupta/Desktop/results/results12/checkpoint.h5')



"""making prediction on full Image"""

"""initialize list for saving the predictions"""
disparity_list = []
if lens_type=='four':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_bottom])
    len(new_predictions)
    
    
for i in range(len(new_predictions)):
    disparity_list.append(new_predictions[i])
len(disparity_list)

""" converting the list in numpy arrays """

disparity_list = asarray(np.array(disparity_list))
disparity_list[0]
len(disparity_list)
"""path for saving the numpy arrays with name disparity_list and .npy as extension and here we need to specify the numpy
array name (in our case name is disparity_list only) that we want to save """

save('/home/rgupta/Desktop/six_lens_result/result3/disparity_list.npy',disparity_list)





