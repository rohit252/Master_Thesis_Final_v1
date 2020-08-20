#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:40:15 2020

@author: rgupta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" import libraries """
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
from keras.utils import plot_model

""" import image handle library """
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


""" argument to run full program from coonsole """

lens_type = sys.argv[1]
print(lens_type)


# setup for learning
EPOCH = 50
BATCH_SIZE = 64
adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=False)


# setup data path for train and prediction 
""" path where data for training and prediction are saved """
data_set_path = "your path" 

""" path where our model weights will be saved after training """
checkpoint_path = '/home/rgupta/Desktop/remote_result/result11/'


""" here we are joining folders inside train and prediction folder to the path for data retrieval"""
if lens_type == 'two':
    
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(training_set_path,'dis_images1')
    train_left_path = os.path.join(training_set_path,'left')
    train_right_path = os.path.join(training_set_path,'right')
    
    prediction_set_path = os.path.join(data_set_path,'prediction')
    pred_left_path = os.path.join(prediction_set_path,'left')
    pred_right_path = os.path.join(prediction_set_path,'right')

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
    


# Load image from folder
def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list=natsort.natsorted(set_list)
#    print(set_list)
    for set_path in set_list:
        img = cv2.imread(os.path.join(folder,set_path),cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        red = img[:,:,2] # extracting the red channel
        all_images.append(red)
    return np.array(all_images)

""" making our fully convolutional neural network model for two-lens Input data"""   

if lens_type == 'two':
    def disparity_cnn_model(input_shape):
        shape=(None, input_shape[1], input_shape[2],input_shape[3])
        left = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        print(left)
        print(right)
        left_1 = Conv2D(filters=32, kernel_size=3,padding='same')(left)
        left_1_pool = MaxPooling2D(2)(left_1)
        left_1_activate = Activation('relu')(left_1_pool)
        
    
        left_2 = Conv2D(filters=62, kernel_size=3,padding='same')(left_1_activate)
        left_2_pool = MaxPooling2D(2)(left_2)
        left_2_activate = Activation('relu')(left_2_pool)
        
    
        left_3 = Conv2D(filters=92, kernel_size=3,padding='same')(left_2_activate)
        left_3_activate = Activation('relu')(left_3)
        
    
        right_1 = Conv2D(filters=32, kernel_size=3,padding='same')(right)
        right_1_pool = MaxPooling2D(2)(right_1)
        right_1_activate = Activation('relu')(right_1_pool)
        
    
        right_2 = Conv2D(filters=62, kernel_size=3,padding='same')(right_1_activate)
        right_2_pool = MaxPooling2D(2)(right_2)
        right_2_activate = Activation('relu')(right_2_pool)
        
    
        right_3 = Conv2D(filters=92, kernel_size=3,padding='same')(right_2_activate)
        right_3_activate = Activation('relu')(right_3)
        
    
        merge = concatenate([left_3_activate, right_3_activate])
    
        merge_1 = Conv2DTranspose(filters=62, kernel_size=3,strides = (2,2), padding='same')(merge)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(merge_1)
        merge_1_activate = Activation('relu')(zero_padd)
        
    
        merge_2 = Conv2DTranspose(filters=22, kernel_size=3,strides=(2,2), padding='same')(merge_1_activate)
        zero_padd1 = ZeroPadding2D(padding=((0,0),(1,0)))(merge_2)
        merge_2_activate = Activation('relu')(zero_padd1)
        
    
        merge_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1),padding='same')(merge_2_activate)
        merge_3_activate = Activation('relu')(merge_3)
    
        model = Model([left, right], merge_3_activate)
    
        return model

    # Load train and prediction set from folder path
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
    pred_left=load_images_from_folder(pred_left_path)
    #plt.imshow(pred_left[10])
    pred_left = np.expand_dims(pred_left,axis = 3)
    print('load pred_right')
    pred_right=load_images_from_folder(pred_right_path)
    #plt.imshow(pred_right[10])
    pred_right = np.expand_dims(pred_right,axis = 3)
    
    print(train_left.shape)
    print(train_right.shape)
    print(train_disp.shape)

    print(pred_left.shape)
    print(pred_right.shape)

 """ making our fully convolutional neural network model for six-lens Input data"""   

if lens_type == 'six':
    def disparity_cnn_model(input_shape):
        shape=(None, input_shape[1], input_shape[2],input_shape[3])
        middle = Input(batch_shape=shape)
        right = Input(batch_shape=shape)
        left = Input(batch_shape=shape)
        top = Input(batch_shape=shape)
        topl = Input(batch_shape=shape)
        bottom =  Input(batch_shape=shape)
        bottoml =  Input(batch_shape=shape)
    
        
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
        
        left_1 = Conv2D(filters=32, kernel_size=3,padding='same')(left)
        left_1_pool = MaxPooling2D(2)(left_1)
        left_1_activate = Activation('relu')(left_1_pool)
        
    
        left_2 = Conv2D(filters=62, kernel_size=3,padding='same')(left_1_activate)
        left_2_pool = MaxPooling2D(2)(left_2)
        left_2_activate = Activation('relu')(left_2_pool)
        
    
        left_3 = Conv2D(filters=92, kernel_size=3,padding='same')(left_2_activate)
        left_3_activate = Activation('relu')(left_3)
        
        top_1 = Conv2D(filters=32, kernel_size=3,padding='same')(top)
        top_1_pool = MaxPooling2D(2)(top_1)
        top_1_activate = Activation('relu')(top_1_pool)
        
    
        top_2 = Conv2D(filters=62, kernel_size=3,padding='same')(top_1_activate)
        top_2_pool = MaxPooling2D(2)(top_2)
        top_2_activate = Activation('relu')(top_2_pool)
        
    
        top_3 = Conv2D(filters=92, kernel_size=3,padding='same')(top_2_activate)
        top_3_activate = Activation('relu')(top_3)
        
        top_l = Conv2D(filters=32, kernel_size=3,padding='same')(topl)
        top_l_pool = MaxPooling2D(2)(top_l)
        top_l_activate = Activation('relu')(top_l_pool)
        
    
        top_l = Conv2D(filters=62, kernel_size=3,padding='same')(top_1_activate)
        top_l_pool = MaxPooling2D(2)(top_l)
        top_l_activate = Activation('relu')(top_l_pool)
        
    
        top_l = Conv2D(filters=92, kernel_size=3,padding='same')(top_l_activate)
        top_l_activate = Activation('relu')(top_l)
        
        
        bottom_1 = Conv2D(filters=32, kernel_size=3,padding='same')(bottom)
        bottom_1_pool = MaxPooling2D(2)(bottom_1)
        bottom_1_activate = Activation('relu')(bottom_1_pool)
        
    
        bottom_2 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_1_activate)
        bottom_2_pool = MaxPooling2D(2)(bottom_2)
        bottom_2_activate = Activation('relu')(bottom_2_pool)
        
    
        bottom_3 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_2_activate)
        bottom_3_activate = Activation('relu')(bottom_3)
        
        bottom_l = Conv2D(filters=32, kernel_size=3,padding='same')(bottoml)
        bottom_l_pool = MaxPooling2D(2)(bottom_l)
        bottom_l_activate = Activation('relu')(bottom_l_pool)
        
    
        bottom_l = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_l_activate)
        bottom_l_pool = MaxPooling2D(2)(bottom_l)
        bottom_l_activate = Activation('relu')(bottom_l_pool)
        
    
        bottom_l = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_l_activate)
        bottom_l_activate = Activation('relu')(bottom_l)
        
    
        merge = concatenate([middle_3_activate,right_3_activate,left_3_activate, top_3_activate,top_l_activate,bottom_3_activate,bottom_l_activate])
    
        merge_1 = Conv2DTranspose(filters=62, kernel_size=3,strides = (2,2), padding='same')(merge)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(merge_1)
        merge_1_activate = Activation('relu')(zero_padd)
        
    
        merge_2 = Conv2DTranspose(filters=22, kernel_size=3,strides=(2,2), padding='same')(merge_1_activate)
        zero_padd1 = ZeroPadding2D(padding=((0,0),(1,0)))(merge_2)
        merge_2_activate = Activation('relu')(zero_padd1)
        
    
        merge_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1),padding='same')(merge_2_activate)
        merge_3_activate = Activation('relu')(merge_3)
    
        model = Model([middle,right,left,top,topl,bottom,bottoml], merge_3_activate)
    
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
        shape=(None, input_shape[1], input_shape[2],input_shape[3])
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
        
        
    
        
        middle_1 = Conv2D(filters=32, kernel_size=3,padding='same')(middle)
        middle_1_pool = MaxPooling2D(2)(middle_1)
        middle_1_activate = Activation('relu')(middle_1_pool)
        
    
        middle_2 = Conv2D(filters=62, kernel_size=3,padding='same')(middle_1_activate)
        middle_2_pool = MaxPooling2D(2)(middle_2)
        middle_2_activate = Activation('relu')(middle_2_pool)
        
    
        middle_3 = Conv2D(filters=92, kernel_size=3,padding='same')(middle_2_activate)
        middle_3_activate = Activation('relu')(middle_3)
        print(middle_3_activate.shape)
        
    
        right_1 = Conv2D(filters=32, kernel_size=3,padding='same')(right)
        right_1_pool = MaxPooling2D(2)(right_1)
        right_1_activate = Activation('relu')(right_1_pool)
        
    
        right_2 = Conv2D(filters=62, kernel_size=3,padding='same')(right_1_activate)
        right_2_pool = MaxPooling2D(2)(right_2)
        right_2_activate = Activation('relu')(right_2_pool)
        
    
        right_3 = Conv2D(filters=92, kernel_size=3,padding='same')(right_2_activate)
        right_3_activate = Activation('relu')(right_3)
        print(right_3_activate.shape)
        
        left_1 = Conv2D(filters=32, kernel_size=3,padding='same')(left)
        left_1_pool = MaxPooling2D(2)(left_1)
        left_1_activate = Activation('relu')(left_1_pool)
        
    
        left_2 = Conv2D(filters=62, kernel_size=3,padding='same')(left_1_activate)
        left_2_pool = MaxPooling2D(2)(left_2)
        left_2_activate = Activation('relu')(left_2_pool)
        
    
        left_3 = Conv2D(filters=92, kernel_size=3,padding='same')(left_2_activate)
        left_3_activate = Activation('relu')(left_3)
        print(left_3_activate.shape)
        
        top_1 = Conv2D(filters=32, kernel_size=3,padding='same')(topl)
        top_1_pool = MaxPooling2D(2)(top_1)
        top_1_activate = Activation('relu')(top_1_pool)
        
    
        top_2 = Conv2D(filters=62, kernel_size=3,padding='same')(top_1_activate)
        top_2_pool = MaxPooling2D(2)(top_2)
        top_2_activate = Activation('relu')(top_2_pool)
        
    
        top_3 = Conv2D(filters=92, kernel_size=3,padding='same')(top_2_activate)
        top_3_activate = Activation('relu')(top_3)
        print(top_3_activate.shape)
        
        top_l1 = Conv2D(filters=32, kernel_size=3,padding='same')(topl1)
        top_l1_pool = MaxPooling2D(2)(top_l1)
        top_l1_activate = Activation('relu')(top_l1_pool)
        
    
        top_l1 = Conv2D(filters=62, kernel_size=3,padding='same')(top_l1_activate)
        top_l1_pool = MaxPooling2D(2)(top_l1)
        top_l1_activate = Activation('relu')(top_l1_pool)
        
    
        top_l1 = Conv2D(filters=92, kernel_size=3,padding='same')(top_l1_activate)
        top_l11_activate = Activation('relu')(top_l1)
        print(top_l11_activate.shape)
        
        top_l2 = Conv2D(filters=32, kernel_size=3,padding='same')(topl2)
        top_l2_pool = MaxPooling2D(2)(top_l2)
        top_l2_activate = Activation('relu')(top_l2_pool)
        
    
        top_l2 = Conv2D(filters=62, kernel_size=3,padding='same')(top_l2_activate)
        top_l2_pool = MaxPooling2D(2)(top_l2)
        top_l2_activate = Activation('relu')(top_l2_pool)
        
    
        top_l2 = Conv2D(filters=92, kernel_size=3,padding='same')(top_l2_activate)
        top_l22_activate = Activation('relu')(top_l2)
        print(top_l22_activate.shape)
        
        top_l3 = Conv2D(filters=32, kernel_size=3,padding='same')(topl3)
        top_l3_pool = MaxPooling2D(2)(top_l3)
        top_l3_activate = Activation('relu')(top_l3_pool)
        
    
        top_l3 = Conv2D(filters=62, kernel_size=3,padding='same')(top_l3_activate)
        top_l3_pool = MaxPooling2D(2)(top_l3)
        top_l3_activate = Activation('relu')(top_l3_pool)
        
    
        top_l3 = Conv2D(filters=92, kernel_size=3,padding='same')(top_l3_activate)
        top_l33_activate = Activation('relu')(top_l3)
        print(top_l33_activate.shape)
        
        top_lr = Conv2D(filters=32, kernel_size=3,padding='same')(toplr)
        top_lr_pool = MaxPooling2D(2)(top_lr)
        top_lr_activate = Activation('relu')(top_lr_pool)
        
    
        top_lr = Conv2D(filters=62, kernel_size=3,padding='same')(top_lr_activate)
        top_lr_pool = MaxPooling2D(2)(top_lr)
        top_lr_activate = Activation('relu')(top_lr_pool)
        
    
        top_lr = Conv2D(filters=92, kernel_size=3,padding='same')(top_lr_activate)
        top_lrr_activate = Activation('relu')(top_l3)
        print(top_lrr_activate.shape)
        
        bottom_1 = Conv2D(filters=32, kernel_size=3,padding='same')(bottoml)
        bottom_1_pool = MaxPooling2D(2)(bottom_1)
        bottom_1_activate = Activation('relu')(bottom_1_pool)
        
    
        bottom_2 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_1_activate)
        bottom_2_pool = MaxPooling2D(2)(bottom_2)
        bottom_2_activate = Activation('relu')(bottom_2_pool)
        
    
        bottom_3 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_2_activate)
        bottom_3_activate = Activation('relu')(bottom_3)
        print(bottom_3_activate.shape)
        
        bottom_l1 = Conv2D(filters=32, kernel_size=3,padding='same')(bottoml1)
        bottom_l1_pool = MaxPooling2D(2)(bottom_l1)
        bottom_l1_activate = Activation('relu')(bottom_l1_pool)
        
    
        bottom_l1 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_l1_activate)
        bottom_l1_pool = MaxPooling2D(2)(bottom_l1)
        bottom_l1_activate = Activation('relu')(bottom_l1_pool)
        
    
        bottom_l1 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_l1_activate)
        bottom_l11_activate = Activation('relu')(bottom_l1)
        print(bottom_l11_activate.shape)
        
        bottom_l2 = Conv2D(filters=32, kernel_size=3,padding='same')(bottoml2)
        bottom_l2_pool = MaxPooling2D(2)(bottom_l2)
        bottom_l2_activate = Activation('relu')(bottom_l2_pool)
        
    
        bottom_l2 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_l2_activate)
        bottom_l2_pool = MaxPooling2D(2)(bottom_l2)
        bottom_l2_activate = Activation('relu')(bottom_l2_pool)
        
    
        bottom_l2 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_l2_activate)
        bottom_l22_activate = Activation('relu')(bottom_l2)
        print(bottom_l22_activate.shape)
        
        bottom_l3 = Conv2D(filters=32, kernel_size=3,padding='same')(bottoml3)
        bottom_l3_pool = MaxPooling2D(2)(bottom_l3)
        bottom_l3_activate = Activation('relu')(bottom_l3_pool)
        
    
        bottom_l3 = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_l3_activate)
        bottom_l3_pool = MaxPooling2D(2)(bottom_l3)
        bottom_l3_activate = Activation('relu')(bottom_l3_pool)
        
    
        bottom_l3 = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_l3_activate)
        bottom_l33_activate = Activation('relu')(bottom_l3)
        print(bottom_l33_activate.shape)
        
        bottom_lr = Conv2D(filters=32, kernel_size=3,padding='same')(bottomlr)
        bottom_lr_pool = MaxPooling2D(2)(bottom_lr)
        bottom_lr_activate = Activation('relu')(bottom_lr_pool)
        
    
        bottom_lr = Conv2D(filters=62, kernel_size=3,padding='same')(bottom_lr_activate)
        bottom_lr_pool = MaxPooling2D(2)(bottom_lr)
        bottom_lr_activate = Activation('relu')(bottom_lr_pool)
        
    
        bottom_lr = Conv2D(filters=92, kernel_size=3,padding='same')(bottom_lr_activate)
        bottom_lrr_activate = Activation('relu')(bottom_lr)
        print(bottom_lrr_activate.shape)
    
        merge = concatenate([middle_3_activate,right_3_activate,left_3_activate, top_3_activate,top_l11_activate,top_l22_activate,top_l33_activate,top_lrr_activate,bottom_3_activate,bottom_l11_activate,bottom_l22_activate,bottom_l33_activate,bottom_lrr_activate])
    
        merge_1 = Conv2DTranspose(filters=62, kernel_size=3,strides = (2,2), padding='same')(merge)
        zero_padd = ZeroPadding2D(padding=((0,0),(1,0)))(merge_1)
        merge_1_activate = Activation('relu')(zero_padd)
        
    
        merge_2 = Conv2DTranspose(filters=22, kernel_size=3,strides=(2,2), padding='same')(merge_1_activate)
        zero_padd1 = ZeroPadding2D(padding=((0,0),(1,0)))(merge_2)
        merge_2_activate = Activation('relu')(zero_padd1)
        
    
        merge_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1),padding='same')(merge_2_activate)
        merge_3_activate = Activation('relu')(merge_3)
    
        model = Model([middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr], merge_3_activate)
    
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
    train_topr.shape
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
    


""" building model"""
model = disparity_cnn_model(train_left.shape)


""" making function for root mean squared error """
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
""" comipling our model """
model.compile(loss='mse',
              optimizer=adam, metrics = [rmse])
"""set early stopping criteria"""
pat = 3 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss',patience=pat, verbose=1)
## Add Learning option and learning
""" here we are saving all the model weights and everything about the model so that we can load out trained model again"""

checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = True)
""" passing path where we want to save our logger , try to save everything in same path as
checkpoint path saved before"""
logger = CSVLogger(filename='/home/rgupta/Desktop/remote_result/result11/log.csv')

""" to see the parameter and details model """
model.summary()

""" fitting the model for learning """
if lens_type == 'two':
    history = model.fit([train_left, train_right],
                        train_disp,
                        epochs = EPOCH,
                        validation_split=0.2,
                        batch_size = BATCH_SIZE,
                        callbacks=[checkpoint,logger,early_stopping])
    
if lens_type == 'six':
    history = model.fit([train_middle,train_left, train_right,train_top,train_top1,train_bottom,train_bottom1],
                    train_disp,
                    epochs = EPOCH,
                    validation_split=0.2,
                    batch_size = BATCH_SIZE,
                    callbacks=[checkpoint,logger,early_stopping])
if lens_type == 'twelve':
    history = model.fit([train_middle,train_left, train_right,train_top,train_top1,train_top2,train_top3,train_topr,train_bottom,train_bottom1,train_bottom2,train_bottom3,train_bottoml],
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
"""path for saving the graph try to use the same path for everything which we are saving """
plt.savefig('/home/rgupta/Desktop/remote_result/result11/train_loss.png')
plt.close()

## draw and save rmse graph

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['train_rmse','val_rmse'],loc = 'best')
plt.grid()
plt.title('model performance')
plt.ylabel('metrics')
plt.xlabel('epoch')
"""" path for saving the graph """
plt.savefig('/home/rgupta/Desktop/remote_result/result11/train_acc.png')
plt.close()

""" if we want to load the weights of our trained model and we can use it for prediction later on different Image 
even we can also trained our already trained model. specify path where our checkpoint.h5 file is saved """
# loading saved model
#model1 = model.load_weights('/home/rgupta/Desktop/results/results12/checkpoint.h5')


#making prediction on full image
""" list for saving the predictions """
disparity_list = []
if lens_type == 'two':
    new_predictions = model.predict([pred_left,pred_right])
if lens_type == 'six':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_bottom,pred_bottom1])
if lens_type == 'twelve':
    new_predictions = model.predict([pred_middle,pred_right,pred_left,pred_top,pred_top1,pred_top2,pred_top3,pred_topr,pred_bottom,pred_bottom1,pred_bottom2,pred_bottom3,pred_bottoml])

"""saving the predictons in a list """
for i in range(len(new_predictions)):
    disparity_list.append(new_predictions[i])

""" converting the list in numpy arrays """
disparity_list = asarray(np.array(disparity_list))
disparity_list[0]
len(disparity_list)
"""path for saving the numpy arrays use the same path for storing everything for simplication purpose """
save('/home/rgupta/Desktop/remote_result/result11/disparity_list.npy',disparity_list)





