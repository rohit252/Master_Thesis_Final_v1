#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:52:43 2020

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
from keras.layers import UpSampling2D ,Cropping2D
from keras.layers import concatenate,Add,Reshape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger,EarlyStopping
from keras.models import load_model, Model, Sequential
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop,Adam


""" importing  image handle library """

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


""" setup for learning """

EPOCH = 70
BATCH_SIZE = 16


""" path where data for training and prediction are saved """
data_set_path = "" 

""" path where our model weights wull be saved after training """

checkpoint_path = ''

""" here we are joining folders inside train and prediction folder to the path for data retrieval"""

if lens_type == 'two':
    training_set_path = os.path.join(data_set_path,'train')
    train_disp_path = os.path.join(train_set_path,'dis_images1')
    train_left_path = os.path.join(train_set_path,'left')
    train_right_path = os.path.join(train_set_path,'right')
    
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
    train_disp_path = os.path.join(data_set_path,'dis_images1')
    train_middle_path = os.path.join(data_set_path,'middle')
    train_right_path = os.path.join(data_set_path,'right')
    train_left_path = os.path.join(data_set_path,'left')
    train_top_path = os.path.join(data_set_path,'top')
    train_top1_path = os.path.join(data_set_path,'top1')
    train_top2_path = os.path.join(data_set_path,'top2')
    train_top3_path = os.path.join(data_set_path,'top3')
    train_topr_path = os.path.join(data_set_path,'top_right')
    train_bottom_path = os.path.join(data_set_path,'bottom')
    train_bottom1_path = os.path.join(data_set_path,'bottom1')
    train_bottom2_path = os.path.join(data_set_path,'bottom2')
    train_bottom3_path = os.path.join(data_set_path,'bottom3')
    train_bottoml_path = os.path.join(data_set_path,'bottom_left')
    
    
    
    
    
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


"""functions that is used in fully convolutional neural model """

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


""" function for root mean squared error loss """
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


""" making our fully convolutional neural network for two-lens input data """

if lens_type == 'two':
    
    def define_epinet(sz_input,sz_input2,conv_depth):

        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
        left = Input(shape=(sz_input,sz_input2,1))
        right = Input(shape=(sz_input,sz_input2,1))
    
      
    
        
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        left_1=layer1_multistream(sz_input,sz_input2,1)(left)
        right_1=layer1_multistream(sz_input,sz_input2,1)(right)
        
        
        ''' Merge layers ''' 
        mid_merged = concatenate([left_1,right_1],  name='mid_merged')
        print(mid_merged)
        tensor_shape= mid_merged.get_shape()
        batch_size=tensor_shape[3]
        print(batch_size)
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input,sz_input2,batch_size)(mid_merged)
        print(mid_merged_)
        
    
        
        ''' Last Conv layer : Conv - Relu - Conv '''
        output=layer3_last(sz_input-18,sz_input2-18,280)(mid_merged_)
        print(output)
    
        
        
        up_1 = Conv2D(filters=62,kernel_size=2,padding='same')(output)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((4,0),(9,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
    
    
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
    
        
        opt = RMSprop(lr=0.1**4)
        model = Model(inputs = [left,right], outputs = [up_2])
        model.compile(optimizer=opt, loss='mse',metrics=[rmse])
    #    model_512.summary() 
        
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
    
    def define_epinet(sz_input,sz_input2,conv_depth):
    
        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
        middle = Input(shape=(sz_input,sz_input2,1))
        right = Input(shape=(sz_input,sz_input2,1))
        left = Input(shape=(sz_input,sz_input2,1))
        topl = Input(shape=(sz_input,sz_input2,1))
        topl1 = Input(shape=(sz_input,sz_input2,1))
        
        bottoml= Input(shape=(sz_input,sz_input2,1))
        bottoml1= Input(shape=(sz_input,sz_input2,1))
      
    
        
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        middle_1=layer1_multistream(sz_input,sz_input2,1)(middle)
        right_1=layer1_multistream(sz_input,sz_input2,1)(right)
        left_1=layer1_multistream(sz_input,sz_input2,1)(left)
        top_l=layer1_multistream(sz_input,sz_input2,1)(topl)
        top_l1=layer1_multistream(sz_input,sz_input2,1)(topl1)
        
        bottom_l=layer1_multistream(sz_input,sz_input2,1)(bottoml)
        bottom_l1=layer1_multistream(sz_input,sz_input2,1)(bottoml1)
        
        ''' Merge layers ''' 
        mid_merged = concatenate([middle_1,right_1,left_1,top_l,top_l1,bottom_l,bottom_l1],  name='mid_merged')
        print(mid_merged)
        tensor_shape= mid_merged.get_shape()
        batch_size=tensor_shape[3]
        
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input,sz_input2,batch_size)(mid_merged)
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
    
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
    
        """ optimizer"""
        opt = RMSprop(lr=0.1**4)
        model = Model(inputs = [middle,right,left,topl,topl1,bottoml,bottoml1], outputs = [up_2])
        """compiling the model"""
        model.compile(optimizer=opt, loss='mse',metrics=[rmse])
        
        return model

    """ Load train and predcition set from folder path """
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
    def define_epinet(sz_input,sz_input2,conv_depth):

        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
        middle = Input(shape=(sz_input,sz_input2,1))
        right = Input(shape=(sz_input,sz_input2,1))
        left = Input(shape=(sz_input,sz_input2,1))
        topl = Input(shape=(sz_input,sz_input2,1))
        topl1 = Input(shape=(sz_input,sz_input2,1))
        topl2 = Input(shape=(sz_input,sz_input2,1))
        topl3 = Input(shape=(sz_input,sz_input2,1))
        toplr = Input(shape=(sz_input,sz_input2,1))
        bottoml= Input(shape=(sz_input,sz_input2,1))
        bottoml1= Input(shape=(sz_input,sz_input2,1))
        bottoml2= Input(shape=(sz_input,sz_input2,1))
        bottoml3= Input(shape=(sz_input,sz_input2,1))
        bottomlr= Input(shape=(sz_input,sz_input2,1))
    
        
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        middle_1=layer1_multistream(sz_input,sz_input2,1)(middle)
        right_1=layer1_multistream(sz_input,sz_input2,1)(right)
        left_1=layer1_multistream(sz_input,sz_input2,1)(left)
        top_l=layer1_multistream(sz_input,sz_input2,1)(topl)
        top_l1=layer1_multistream(sz_input,sz_input2,1)(topl1)
        top_l2=layer1_multistream(sz_input,sz_input2,1)(topl2)
        top_l3=layer1_multistream(sz_input,sz_input2,1)(topl3)
        top_lr=layer1_multistream(sz_input,sz_input2,1)(toplr)
        bottom_l=layer1_multistream(sz_input,sz_input2,1)(bottoml)
        bottom_l1=layer1_multistream(sz_input,sz_input2,1)(bottoml1)
        bottom_l2=layer1_multistream(sz_input,sz_input2,1)(bottoml2)
        bottom_l3=layer1_multistream(sz_input,sz_input2,1)(bottoml3)
        bottom_lr=layer1_multistream(sz_input,sz_input2,1)(bottomlr)
    
    
    
    
    
    
        
    
        ''' Merge layers ''' 
        mid_merged = concatenate([middle_1,right_1,left_1,top_l,top_l1,top_l2,top_l3,top_lr,bottom_l,bottom_l1,bottom_l2,bottom_l3,bottom_lr],  name='mid_merged')
        print(mid_merged)
        tensor_shape= mid_merged.get_shape()
        batch_size=tensor_shape[3]
        print(batch_size)
        
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input,sz_input2,batch_size)(mid_merged)
        print(mid_merged_)
    

        
        ''' Last Conv layer : Conv - Relu - Conv '''
        output=layer3_last(sz_input-18,sz_input2-18,280)(mid_merged_)
        print(output)
    
    
        
        up_1 = Conv2D(filters=62,kernel_size=2,padding='same')(output)
        up_1 = UpSampling2D(2)(up_1)
        zero_padd = ZeroPadding2D(padding=((4,0),(9,0)))(up_1)
        up_1 = Activation('relu')(zero_padd)
        
        up_1_batch=BatchNormalization(axis=3)(up_1)
    
        up_2 = Conv2D(filters=1,kernel_size=3,padding='same')(up_1_batch)
        up_2=Activation('relu')(up_2)
    
        
        opt = RMSprop(lr=0.1**4)
        model = Model(inputs = [middle,right,left,topl,topl1,topl2,topl3,toplr,bottoml,bottoml1,bottoml2,bottoml3,bottomlr], outputs = [up_2])
        model.compile(optimizer=opt, loss='mse',metrics=[rmse])
    #    model_512.summary() 
        
        return model

  """ Load train and prediction set from folder path """
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

    
""" number of layers """
conv_depth=7
""" building the model """
model = define_epinet(40,35,
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
logger = CSVLogger(filename='/home/rgupta/Desktop/six_lens_result/epinet/log.csv')
""" to see the layers,number of parameters and details of the model """                

model.summary()
""" fitting the model for training """

if lens_type == 'two':
    
    history = model.fit([train_left,train_right],
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

## plotting loss ##
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss','val_loss'],loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
"""path for saving the graph with name train_loss and .png as extension 
we have used same folder to save everything which we need to save """
plt.savefig('/home/rgupta/Desktop/six_lens_result/epinet/train_loss.png')
plt.close()

## rmse graph 

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.grid()
plt.title('model performance')
plt.ylabel('metrics')
plt.xlabel('epoch')
""" saving the graph with train_acc name with .png as extension """

plt.savefig('/home/rgupta/Desktop/six_lens_result/epinet/train_rmse.png')
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



save('/home/rgupta/Desktop/six_lens_result/epinet/disparity_list.npy',disparity_list)