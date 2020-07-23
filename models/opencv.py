#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:41:03 2020

@author: rgupta
"""


from __future__ import division


"""importing libarries """
import natsort
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save

""" path for taking all left lens files"""

path1 = '/home/rgupta/Desktop/two_lenses/prediction1/left'
"""list for taking all files """
path_left = os.listdir(path1)
"""sorting all files in a list"""

path_left = natsort.natsorted(path_left)

""" path for taking all right lens files"""

path2 =   '/home/rgupta/Desktop/two_lenses/prediction1/right'
"""list for taking all files """

path_right = os.listdir(path2)
"""sorting all files in a list"""
path_right = natsort.natsorted(path_right)

"""initialize list for storing all left and right lens/data"""

img_l=[]
img_r=[]

"""reading all left lenses"""
for left in path_left:
    fullpath = os.path.join(path1,left)
    img = cv2.imread(fullpath)
    img = img[:,:,2]
    img_l.append(img)
    
"""reading all right lenses """  

for right in path_right:
    fullpath = os.path.join(path2,right)
    img = cv2.imread(fullpath)
    img = img[:,:,2]
    img_r.append(img)

    
left= img_l[0]

""" taking height and width of an lens """
h,w=left.shape


""" implementing streo block matching"""
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

padding = 16
count=0


""" Initialize list for saving predictions """

disparity=[]
for i in range(len(img_l)):
    pad_imgL = np.concatenate((np.zeros((h,padding),np.uint8),img_l[count]),1)
    pad_imgR = np.concatenate((np.zeros((h,padding),np.uint8),img_r[count]),1)
    disp = stereo.compute(pad_imgL,pad_imgR)[:,padding:].astype(np.float32) / 16.0
    disparity.append(disp)
    count+=1
    
"""saving the prediction result in a list using numpy """

disparity_list = asarray(np.array(disparity))
disparity_list[0]
len(disparity_list)
"""path for saving the numpy arrays with name disparity_list and .npy as extension and here we need to specify the numpy
array name (in our case name is disparity_list only) that we want to save """


save('/home/rgupta/Desktop/two_lens_result_new/stereo/disparity_list.npy',disparity_list)
    
    
    
    
    
    
