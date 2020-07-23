# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:15:49 2020

@author: Home
"""
"""Importing Libraries"""
import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import load
import natsort


""" path for extracting rectangular cutouts """
path='/home/rgupta/Desktop/original_disparity/disparity_9_two'

""" width and height of our hexagonal Input data/lens"""
width_required=35
height_required=40


""" We are creating a blank BGR Image of size 3500 by 3500 by  creating function create_blank """

def create_blank(width, height, rgb_color=(0,0,0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
#     a = [np.zeros((3500,3500,3)).astype(object), np.zeros((3500,3500,3)).astype(object), np.zeros((3500,3500,3)).astype(object)]
    """ converting image in BGR format"""
    color = tuple(reversed(rgb_color))
#    image[:] = color
    image[:] = color

    return image

width1, height1 = 3500, 3500

all = (0,0,0)
image = create_blank(width1, height1, rgb_color=all)

""" We just need the red channel values so we are extracting only red channel """

image = image[:,:,2]

"""For generating our predicted Image we have to expand the dimension of our created blank Image"""

image=np.expand_dims(image,axis=2)
print(image.shape)

filelist = os.listdir(path)

"""sorting out the cutouts that we have extracted """

dis = natsort.natsorted(filelist)

"""path for extracting the coordinates of the cutouts """

coor_values = load('/home/rgupta/Desktop/two_lens_result/resnet/coor_values.npy')
counter=0


""" As we know that we have used hexagonal lenses so to see the output without overlapping and each hexagonal shape we have created 
small hexagonal Image to ensure that we do not see any overlapping in our output predicted Image, loading that Image"""

small_image=load('/home/rgupta/Desktop/small_image.npy')

"""we are just interested in red channel values, so just extracting red cahnnel"""

small_image = small_image[:,:,2]

""" expanding the dimension for our work"""

small_image = np.expand_dims(small_image,axis=2)
print(small_image.shape)

""" In this for loop we are doing our work, we are taking all the rectangular cutouts
and we are simply pasting our hexagonal disparity patches to the exact coordinates in order to see the 
same Input Image that we have given to the network for prediction and then by analysing our predicted image
and we are getting our hexagonal structure by multiplying rectangular patches with our hexagonal small patch that we have created""" 


for file in dis:
    dis_fullpath = os.path.join(path,file)
    x,y = coor_values[counter]
    a1 = cv2.imread(dis_fullpath,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    a1=a1[:,:,2]
    a1=np.expand_dims(a1,axis=2)
    final=np.multiply(a1,small_image)
    image[int(y):int(y)+height_required,int(x):int(x)+width_required]= image[int(y):int(y)+height_required,int(x):int(x)+width_required]+final
    counter+=1


"""path for storing the ground truth Image consisting of hexagonal cutouts """
cv2.imwrite('/home/rgupta/Desktop/original_images_pred/orgtwo9.exr',image)

