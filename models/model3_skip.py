#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:59:05 2020

@author: rgupta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" import keras libraries """
import keras
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D 
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger,EarlyStopping
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

""" import image handle library """
import numpy as np
import natsort
import cv2
import math
from keras import backend as K
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
from numpy import load
from numpy import asarray
from numpy import save


"""argument to run the whole program from console"""


lens_type = sys.argv[1]
print(lens_type)

""" setup for learning """
EPOCH = 70
BATCH_SIZE = 64
adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=False)




""" path where data for training and prediction are saved """
data_set_path = "" 

""" path where our model weights wull be saved after training """

checkpoint_path = ''

""" here we are joining folders inside train and prediction folder to the path for data retrieval"""


if lens_type == 'six':
    
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_middle_path = os.path.join(training_set_path,'middle')
    train_right_path = os.path.join(training_set_path,'right')
    train_left_path = os.path.join(training_set_path,'left')
    train_top_path = os.path.join(training_set_path,'top')
    train_top1_path = os.path.join(training_set_path,'top1')
    train_bottom_path = os.path.join(training_set_path,'bottom')
    train_bottom1_path = os.path.join(training_set_path,'bottom1')
    
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_middle_path = os.path.join(prediction_set_path,'middle')
    pred_right_path = os.path.join(prediction_set_path,'right')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_top_path = os.path.join(prediction_set_path,'top')
    pred_top1_path = os.path.join(prediction_set_path,'top1')
    pred_bottom_path = os.path.join(prediction_set_path,'bottom')
    pred_bottom1_path = os.path.join(prediction_set_path,'bottom1')

if lens_type == 'twelve':

    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_middle_path = os.path.join(training_set_path,'middle')
    train_right_path = os.path.join(training_set_path,'right')
    train_left_path = os.path.join(training_set_path,'left')
    train_top_path = os.path.join(training_set_path,'top')
    train_top1_path = os.path.join(training_set_path,'top1')
    train_top2_path = os.path.join(training_set_path,'top2')
    train_top3_path = os.path.join(training_set_path,'top3')
    train_topr_path = os.path.join(training_set_path,'top right')
    train_bottom_path = os.path.join(training_set_path,'bottom')
    train_bottom1_path = os.path.join(training_set_path,'bottom1')
    train_bottom2_path = os.path.join(training_set_path,'bottom2')
    train_bottom3_path = os.path.join(training_set_path,'bottom3')
    train_bottoml_path = os.path.join(training_set_path,'bottom left')
    
    
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_middle_path = os.path.join(prediction_set_path,'middle')
    pred_right_path = os.path.join(prediction_set_path,'right')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_top_path = os.path.join(prediction_set_path,'top')
    pred_top1_path = os.path.join(prediction_set_path,'top1')
    pred_top2_path = os.path.join(prediction_set_path,'top2')
    pred_top3_path = os.path.join(prediction_set_path,'top3')
    pred_topr_path = os.path.join(prediction_set_path,'top_right')
    pred_bottom_path = os.path.join(prediction_set_path,'bottom')
    pred_bottom1_path = os.path.join(prediction_set_path,'bottom1')
    pred_bottom2_path = os.path.join(prediction_set_path,'bottom2')
    pred_bottom3_path = os.path.join(prediction_set_path,'bottom3')
    pred_bottoml_path = os.path.join(prediction_set_path,'bottom_left')

    

""" Load image from folder """
def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list = natsort.natsorted(set_list)
    for set_path in set_list:
        img = cv2.imread(os.path.join(folder,set_path),cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        red = img[:,:,2]#extracting red channel
        all_images.append(red)
    return np.array(all_images)

""" making our fully convolutional neural network model for six-lens Input data"""  

if lens_type == 'six':
    def disparity_cnn_model(input_shape):
        shape=(None, input_shape[1], input_shape[2], input_shape[3])
        middle= Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        top = Input(batch_shape=shape)
        top1 = Input(batch_shape=shape)
        bottom = Input(batch_shape=shape)
        bottom1 = Input(batch_shape=shape)
    
        
        merge=concatenate([middle,right,left,top,top1,bottom,bottom1])
        
        conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(merge)
        conv1 = BatchNormalization()(conv1)
        conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
    
        up61 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
        zero_padd = ZeroPadding2D(padding=((1,0),(0,0)))(up61)
        up6 = concatenate([zero_padd, conv4])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
    
        up71 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = concatenate([up71, conv3])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
    
        up81 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        zero_padd = ZeroPadding2D(padding=((0,0),(0,1)))(up81)
        up8 = concatenate([zero_padd, conv2])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
    
        up91 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up91)
        up9 = concatenate([zero_padd, conv1])
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = SeparableConv2D(1, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
    
    
        model = Model([middle,right,left,top,top1,bottom,bottom1], conv9)
    
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
    print('Load train_top1')
    train_top1 = load_images_from_folder(train_top1_path)
    train_top1 = np.expand_dims(train_top1, axis=3)
    print('Load train_bottom')
    train_bottom = load_images_from_folder(train_bottom_path)
    train_bottom = np.expand_dims(train_bottom, axis=3)
    print('Load train_bottom1')
    train_bottom1 = load_images_from_folder(train_bottom1_path)
    train_bottom1 = np.expand_dims(train_bottom1, axis=3)
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
    print('load pred_top1')
    pred_top1=load_images_from_folder(pred_top1_path)
    #plt.imshow(pred_right[10])
    pred_top1 = np.expand_dims(pred_top1,axis = 3)
    print('load pred_bottom')
    pred_bottom=load_images_from_folder(pred_bottom_path)
    #plt.imshow(pred_right[10])
    pred_bottom = np.expand_dims(pred_bottom,axis = 3)
    print('load pred_bottom1')
    pred_bottom1=load_images_from_folder(pred_bottom1_path)
    #plt.imshow(pred_right[10])
    pred_bottom1 = np.expand_dims(pred_bottom1,axis = 3)
    
    print(train_middle.shape)
    print(train_left.shape)
    print(train_right.shape)
    print(train_top.shape)
    print(train_top1.shape)
    print(train_bottom.shape)
    print(train_bottom1.shape)
    print(train_disp.shape)

    print(pred_middle.shape)
    print(pred_top.shape)
    print(pred_top1.shape)
    print(pred_bottom.shape)
    print(pred_bottom1.shape)
    print(pred_left.shape)
    print(pred_right.shape)
 
    
""" making our fully convolutional neural network model for twelve-lens Input data"""  
   
if lens_type == 'twelve':
    def disparity_cnn_model(input_shape):
        shape=(None, input_shape[1], input_shape[2], input_shape[3])
        middle = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        topl = Input(batch_shape=shape)
        topl1 = Input(batch_shape=shape)
        topl2 = Input(batch_shape=shape)
        topl3 = Input(batch_shape=shape)
        toplr = Input(batch_shape=shape)
        bottoml =  Input(batch_shape=shape)
        bottoml1 =  Input(batch_shape=shape)
        bottoml2 =  Input(batch_shape=shape)
        bottoml3 =  Input(batch_shape=shape)
        bottomlr =  Input(batch_shape=shape)
    
        
        merge=concatenate([middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr])
        
        conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(merge)
        conv1 = BatchNormalization()(conv1)
        conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
    
        up61 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
        zero_padd = ZeroPadding2D(padding=((1,0),(0,0)))(up61)
        up6 = concatenate([zero_padd, conv4])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
    
        up71 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = concatenate([up71, conv3])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
    
        up81 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        zero_padd = ZeroPadding2D(padding=((0,0),(0,1)))(up81)
        up8 = concatenate([zero_padd, conv2])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
    
        up91 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up91)
        up9 = concatenate([zero_padd, conv1])
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = SeparableConv2D(1, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
    
    
        model = Model([middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr], conv9)
    
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
    print('Load train_top1')
    train_top1 = load_images_from_folder(train_top1_path)
    train_top1 = np.expand_dims(train_top1, axis=3)
    print('Load train_top2')
    train_top2 = load_images_from_folder(train_top2_path)
    train_top2 = np.expand_dims(train_top2, axis=3)
    print('Load train_top3')
    train_top3 = load_images_from_folder(train_top3_path)
    train_top3 = np.expand_dims(train_top3, axis=3)
    print('Load train_topr')
    train_topr = load_images_from_folder(train_topr_path)
    train_topr = np.expand_dims(train_topr, axis=3)
    print('Load train_bottom')
    train_bottom = load_images_from_folder(train_bottom_path)
    train_bottom = np.expand_dims(train_bottom, axis=3)
    print('Load train_bottom1')
    train_bottom1 = load_images_from_folder(train_bottom1_path)
    train_bottom1 = np.expand_dims(train_bottom1, axis=3)
    print('Load train_bottom2')
    train_bottom2 = load_images_from_folder(train_bottom2_path)
    train_bottom2 = np.expand_dims(train_bottom2, axis=3)
    print('Load train_bottom3')
    train_bottom3 = load_images_from_folder(train_bottom3_path)
    train_bottom3 = np.expand_dims(train_bottom3, axis=3)
    print('Load train_bottoml')
    train_bottoml = load_images_from_folder(train_bottoml_path)
    train_bottoml = np.expand_dims(train_bottoml, axis=3)
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
    print('load pred_top1')
    pred_top1=load_images_from_folder(pred_top1_path)
    #plt.imshow(pred_right[10])
    pred_top1 = np.expand_dims(pred_top1,axis = 3)
    print('load pred_top2')
    pred_top2=load_images_from_folder(pred_top2_path)
    #plt.imshow(pred_right[10])
    pred_top2 = np.expand_dims(pred_top2,axis = 3)
    print('load pred_top3')
    pred_top3=load_images_from_folder(pred_top3_path)
    #plt.imshow(pred_right[10])
    pred_top3 = np.expand_dims(pred_top3,axis = 3)
    print('load pred_topr')
    pred_topr=load_images_from_folder(pred_topr_path)
    #plt.imshow(pred_right[10])
    pred_topr = np.expand_dims(pred_topr,axis = 3)
    print('load pred_bottom')
    pred_bottom=load_images_from_folder(pred_bottom_path)
    #plt.imshow(pred_right[10])
    pred_bottom = np.expand_dims(pred_bottom,axis = 3)
    
    print('load pred_bottom1')
    pred_bottom1=load_images_from_folder(pred_bottom1_path)
    #plt.imshow(pred_right[10])
    pred_bottom1 = np.expand_dims(pred_bottom1,axis = 3)
    print('load pred_bottom2')
    pred_bottom2=load_images_from_folder(pred_bottom2_path)
    #plt.imshow(pred_right[10])
    pred_bottom2 = np.expand_dims(pred_bottom2,axis = 3)
    print('load pred_bottom3')
    pred_bottom3=load_images_from_folder(pred_bottom3_path)
    #plt.imshow(pred_right[10])
    pred_bottom3 = np.expand_dims(pred_bottom3,axis = 3)
    print('load pred_bottoml')
    pred_bottoml=load_images_from_folder(pred_bottoml_path)
    #plt.imshow(pred_right[10])
    pred_bottoml = np.expand_dims(pred_bottoml,axis = 3)
    
    print(train_middle.shape)
    print(train_left.shape)
    print(train_right.shape)
    print(train_top.shape)
    print(train_top1.shape)
    print(train_top2.shape)
    print(train_top3.shape)
    print(train_topr.shape)
    print(train_bottom.shape)
    print(train_bottom1.shape)
    print(train_bottom2.shape)
    print(train_bottom3.shape)
    print(train_bottoml.shape)
    print(train_disp.shape)
    

    print(pred_middle.shape)
    print(pred_top.shape)
    print(pred_top1.shape)
    print(pred_top2.shape)
    print(pred_top3.shape)
    print(pred_topr.shape)
    
    print(pred_bottom.shape)
    print(pred_bottom1.shape)
    print(pred_bottom2.shape)
    print(pred_bottom3.shape)
    print(pred_bottoml.shape)
    
    print(pred_left.shape)
    print(pred_right.shape)


""" function of root mean squared error """ 
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))



"""build model"""
model = disparity_cnn_model(train_left.shape)
""" compiling the model """
model.compile(loss='mse',
              optimizer=adam,metrics=[rmse])

"""set early stopping criteria"""
pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping=EarlyStopping(monitor='val_loss',patience=pat,verbose=1)

## Add Learning option and learning
""" here we are saving all the model weights and everything about the model so that we can load out trained model again"""

checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = True)

""" passing path where we want to save our logger , try to save everything in same path pass same path as
checkpoint path passed before and save the logger with name log and extension .csv"""
logger = CSVLogger(filename='/home/rgupta/Desktop/six_lens_result/result6/log.csv')

""" to see the layers,number of parameters and details of the model """                

model.summary()

""" fitting the model for training """
if lens_type == 'six':
    history = model.fit([train_middle, train_right,train_left,train_top,train_top1,train_bottom,train_bottom1],
                    train_disp,
                    epochs = EPOCH,
                    batch_size = BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[checkpoint,logger,early_stopping])
if lens_type == 'twelve':
    history = model.fit([train_middle, train_right,train_left,train_top,train_top1,train_top2,train_top3,train_topr,train_bottom,train_bottom1,train_bottom2,train_bottom3,train_bottoml],
                    train_disp,
                    epochs = EPOCH,
                    batch_size = BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[checkpoint,logger,early_stopping])

""" graphs to check the training and validation progress of model """

## training_graph ##
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss','val_loss'],loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
"""path for saving the graph with name train_loss and .png as extension 
we have used same folder to save everything which we need to save """

plt.savefig('/home/rgupta/Desktop/six_lens_result/result6/train_loss.png')
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
plt.savefig('/home/rgupta/Desktop/six_lens_result/result6/train_rmse.png')
plt.close()



""" if we want to load the weights of our trained model and we can use it for prediction later on different Image 
even we can also trained our already trained model. specify path where our checkpoint.h5 file is saved, just
cahnge the path and the name of our checkpoint file is checkpoint.h5 so we do not need to change it """
# loading saved model
#model1 = model.load_weights('/home/rgupta/Desktop/results/results12/checkpoint.h5')




"""making prediction on full Image"""

""" Initialize list for saving predictions """
disparity_list = []
if lens_type == 'six':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_bottom,pred_bottom1])
if lens_type == 'twelve':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_top2,pred_top3,pred_topr,pred_bottom,pred_bottom1,pred_bottom2,pred_bottom3,pred_bottoml])

for i in range(len(new_predictions)):
    disparity_list.append(new_predictions[i])
len(disparity_list)

""" converting the list in numpy arrays """


disparity_list = asarray(np.array(disparity_list))
len(disparity_list)

"""path for saving the numpy arrays with name disparity_list and .npy as extension and here we need to specify the numpy
array name (in our case name is disparity_list only) that we want to save """

save('/home/rgupta/Desktop/six_lens_result/result6/disparity_list.npy',disparity_list)
