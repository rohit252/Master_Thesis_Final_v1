#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:03:11 2020

@author: rgupta
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""import keras libraries"""
import keras
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D 
from keras.layers import concatenate,Add
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger,EarlyStopping
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.utils import plot_model
from keras import backend as K


"""import image handle library"""
import numpy as np
import cv2
import argparse
import os
import sys
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




""" setup for learning """
EPOCH = 50
BATCH_SIZE = 64
adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=False)



""" path where data for training and prediction are saved """
data_set_path = "" 

""" path where our model weights wull be saved after training """

checkpoint_path = ''

""" here we are joining folders inside train and prediction folder to the path for data retrieval"""

if lens_type == 'two':
    

    
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_left_path = os.path.join(training_set_path,'left')
    train_right_path = os.path.join(training_set_path,'right')
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_right_path = os.path.join(prediction_set_path,'right')

if lens_type == 'four':
    
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_middle_path = os.path.join(training_set_path,'middle')
    train_right_path = os.path.join(training_set_path,'right')
    train_left_path = os.path.join(training_set_path,'left')
    train_top_path = os.path.join(training_set_path,'top')
    train_bottom_path = os.path.join(training_set_path,'bottom')
    
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_middle_path = os.path.join(prediction_set_path,'middle')
    pred_right_path = os.path.join(prediction_set_path,'right')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_top_path = os.path.join(prediction_set_path,'top')
    pred_bottom_path = os.path.join(prediction_set_path,'bottom')

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
    train_topr_path = os.path.join(training_set_path,'top_right')
    train_bottom_path = os.path.join(training_set_path,'bottom')
    train_bottom1_path = os.path.join(training_set_path,'bottom1')
    train_bottom2_path = os.path.join(training_set_path,'bottom2')
    train_bottom3_path = os.path.join(training_set_path,'bottom3')
    train_bottoml_path = os.path.join(training_set_path,'bottom_left')
    
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
    
    
""" Load image from folder"""
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

def identity_block(X, f, filters, stage, block):

    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):

    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'same')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

""" making our fully convolutional neural network model for two-lens Input data"""  

if lens_type == 'two':
    def disparity_cnn_model(input_shape):
    
        # Define the input as a tensor with shape input_shape
        shape=(None, input_shape[1], input_shape[2], input_shape[3])
        left = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        
        merge =concatenate([left,right])
        
        # Stage 1
        X_l = Conv2D(64, (3, 3), strides=(1, 1), padding = 'same')(merge)
        X_l = BatchNormalization()(X_l)
        X_l = Activation('relu')(X_l)
        X_l = MaxPooling2D((2, 2))(X_l)
    
        # Stage 2
        X_l = convolutional_block(X_l, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='b')
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='c')
        
        # Stage 3 (≈4 lines)
        X_l = convolutional_block(X_l, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 1)
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='b')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='c')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='d')
        
        
        ### upsampling ###
        up_1 = Conv2D(filters=62,kernel_size=3,padding='same')(X_l)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
        
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
        
       # Create model
        model = Model([left,right],up_2)
    
        return model
    
 """ Load train and prediction set from folder path """
    print('Load train_left')
    train_left = load_images_from_folder(train_left_path)
    train_left = np.expand_dims(train_left, axis=3)
    print('Load train_right')
    train_right = load_images_from_folder(train_right_path)
    train_right = np.expand_dims(train_right, axis=3)
    print('Load train_disp')
    train_disp = load_images_from_folder(train_disp_path)
    train_disp = np.expand_dims(train_disp, axis=3)

    print('load pred_left')
    pred_left = load_images_from_folder(pred_left_path)
    #plt.imshow(pred_left[10])
    pred_left = np.expand_dims(pred_left,axis=3)
    print('load pred_right')
    pred_right = load_images_from_folder(pred_right_path)
    #plt.imshow(pred_right[10])
    pred_right = np.expand_dims(pred_right,axis = 3)
    
    print(train_left.shape)
    print(train_right.shape)
    print(train_disp.shape)

    print(pred_left.shape)
    print(pred_right.shape)
 
 """ making our fully convolutional neural network model for four-lens Input data"""  
   
if lens_type == 'four':
    def disparity_cnn_model(input_shape):
        # Define the input as a tensor with shape input_shape
        shape=(None, input_shape[1], input_shape[2], input_shape[3])
        middle = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        top = Input(batch_shape=shape)
        bottom =  Input(batch_shape=shape)

    
        merge =concatenate([middle,right,left,top,bottom])
        
        # Stage 1
        X_l = Conv2D(64, (3, 3), strides=(1, 1), padding = 'same')(merge)
        X_l = BatchNormalization()(X_l)
        X_l = Activation('relu')(X_l)
        X_l = MaxPooling2D((2, 2))(X_l)
    
        # Stage 2
        X_l = convolutional_block(X_l, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='b')
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='c')
        
        # Stage 3 (≈4 lines)
        X_l = convolutional_block(X_l, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 1)
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='b')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='c')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='d')
        
        
        ### upsampling ###
        up_1 = Conv2D(filters=62,kernel_size=3,padding='same')(X_l)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
        
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
        
       # Create model
        model = Model([middle,right,left,top,bottom],up_2)
    
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
    #plt.imshow(pred_right[10])
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

 """ making our fully convolutional neural network model for six-lens Input data"""  

if lens_type == 'six':
    def disparity_cnn_model(input_shape):

        # Define the input as a tensor with shape input_shape
        shape=(None, input_shape[1], input_shape[2], input_shape[3])
        middle = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        top = Input(batch_shape=shape)
        top1=Input(batch_shape=shape)
        bottom =  Input(batch_shape=shape)
        bottom1 =  Input(batch_shape=shape)
    
        
        merge =concatenate([middle,right,left,top,top1,bottom,bottom1])
        
        # Stage 1
        X_l = Conv2D(64, (3, 3), strides=(1, 1), padding = 'same')(merge)
        X_l = BatchNormalization()(X_l)
        X_l = Activation('relu')(X_l)
        X_l = MaxPooling2D((2, 2))(X_l)
    
        # Stage 2
        X_l = convolutional_block(X_l, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='b')
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='c')
        
        # Stage 3 (≈4 lines)
        X_l = convolutional_block(X_l, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 1)
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='b')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='c')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='d')
        
        
        ### upsampling ###
        up_1 = Conv2D(filters=62,kernel_size=3,padding='same')(X_l)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
        
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
        
       # Create model
        model = Model([middle,right,left,top,top1,bottom,bottom1],up_2)
    
        return model
    
    # Load train and test set from folder path
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
    print(test_middle.shape)
    print(test_top.shape)
    print(test_top1.shape)
    print(test_bottom.shape)
    print(test_bottom1.shape)
    print(test_left.shape)
    print(test_right.shape)
    print(test_disp.shape)
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

        # Define the input as a tensor with shape input_shape
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
        
    
        
        merge =concatenate([middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr])
        
        # Stage 1
        X_l = Conv2D(64, (3, 3), strides=(1, 1), padding = 'same')(merge)
        X_l = BatchNormalization()(X_l)
        X_l = Activation('relu')(X_l)
        X_l = MaxPooling2D((2, 2))(X_l)
    
        # Stage 2
        X_l = convolutional_block(X_l, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='b')
        X_l = identity_block(X_l, 3, [64, 64, 256], stage=2, block='c')
        
        # Stage 3 (≈4 lines)
        X_l = convolutional_block(X_l, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 1)
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='b')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='c')
        X_l = identity_block(X_l, 3, [128, 128, 512], stage=3, block='d')
        
        
        ### upsampling ###
        up_1 = Conv2D(filters=62,kernel_size=3,padding='same')(X_l)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
        
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
        
       # Create model
        model = Model([middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr],up_2)
    
        return model
    
    """ Load train and preciction set from folder path """
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
     

""" build model """
model = disparity_cnn_model(train_left.shape)


""" function of root mean squared error """ 

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

""" compiling the model """

model.compile(loss='mse',
              optimizer=adam,metrics=[rmse])


"""set early stopping criteria"""
pat=3#this is the number of epochs with no improvment after which the training will stop
early_stopping=EarlyStopping(monitor='val_loss',patience=pat,verbose=1)
## Add Learning option and learning

""" here we are saving all the model weights and everything about the model so that we can load out trained model again"""

checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = True)

""" passing path where we want to save our logger , try to save everything in same path pass same path as
checkpoint path passed before and save the logger with name log and extension .csv"""

logger = CSVLogger(filename='/home/rgupta/Desktop/try_results1/resnet/log.csv')


""" to see the layers,number of parameters and details of the model """                
model.summary()

""" fitting the model for training """

if lens_type == 'two':
    history = model.fit([train_left, train_right],
                    train_disp,
                    epochs = EPOCH,
                    batch_size = BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[checkpoint,logger,early_stopping])
    
if lens_type == 'four':
    history = model.fit([train_middle,train_right,train_left,train_top,train_bottom],
                    train_disp,
                    epochs = EPOCH,
                    batch_size = BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[checkpoint,logger,early_stopping])

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

# draw and save result

## plotting loss ##
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss','val_loss'],loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
"""path for saving the graph with name train_loss and .png as extension 
we have used same folder to save everything which we need to save """
plt.savefig('/home/rgupta/Desktop/try_results/resnet/train_loss.png')
plt.close()

#### plotting rmse ####

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['train_rmse','val_rmse'],loc = 'best')
plt.grid()
plt.title('model performance')
plt.ylabel('metrics')
plt.xlabel('epoch')

""" saving the graph with train_acc name with .png as extension """

plt.savefig('/home/rgupta/Desktop/try_results/resnet/train_rmse.png')
plt.close()



""" if we want to load the weights of our trained model and we can use it for prediction later on different Image 
even we can also trained our already trained model. specify path where our checkpoint.h5 file is saved, just
cahnge the path and the name of our checkpoint file is checkpoint.h5 so we do not need to change it """
#model.load_weights('/home/rgupta/Desktop/results/checkpoint.h5')


"""making prediction on full Image"""

""" Initialize list for saving predictions """
disparity_list = []
if lens_type == 'two':
    new_predictions = model.predict([pred_left,pred_right])
    
if lens_type == 'four':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_bottom])

if lens_type == 'six':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_bottom,pred_bottom1])

if lens_type == 'twelve':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_top2,pred_top3,pred_topr,pred_bottom,pred_bottom1,pred_bottom2,pred_bottom3,pred_bottoml])
    
len(new_predictions)
for i in range(len(new_predictions)):
    disparity_list.append(new_predictions[i])
len(disparity_list)


""" converting the list in numpy arrays """

disparity_list = asarray(np.array(disparity_list))
len(disparity_list)


"""path for saving the numpy arrays with name disparity_list and .npy as extension and here we need to specify the numpy
array name (in our case name is disparity_list only) that we want to save """

save('/home/rgupta/Desktop/try_results/resnet/disparity_list.npy',disparity_list)







