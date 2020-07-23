# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:42:26 2020

@author: Home
"""

import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import load


"""width and height of our hexagonal Input data"""

width_required=35
height_required=40


"""Initialize counter"""
counter=0


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



"""Extracting coordinate values of every Input lens/data that we have saved while creating 
prediction data and using these coordinate values we will again create full Image using the disparity values 
that our model has given to us"""


coor_values = load('/home/rgupta/Desktop/two_lens_result_new/resnet3/coor.npy')
len(coor_values)

""" Extracting the disparity values of every lens that our model has predicted """


disparity_list = load('/home/rgupta/Desktop/two_lens_result_new/stereo/disparity_list.npy')
print(disparity_list)
len(disparity_list)

""" As we know that we have used hexagonal lenses so to see the output without overlapping and each hexagonal shape we have created 
small hexagonal Image to ensure that we do not see any overlapping in our output predicted Image, loading that Image"""

small_image=load('/home/rgupta/Desktop/small_image.npy')

"""we are just interested in red channel values, so just extracting red cahnnel"""


small_image = small_image[:,:,2]
print(small_image.shape)

""" In this for loop we are doing our work, we are taking all the disparity values that our model has predicted
and we are simply pasting our hexagonal disparity patches to the exact coordinates in order to see the 
same Input Image that we have given to the network for prediction and then by analysing our predicted image
and we are getting our hexagonal structure by multiplying disparity value patch with our hexagonal small patch that we have created""" 

for t in range(len(coor_values)):
    x,y = coor_values[t]
    a1 = disparity_list[counter]
    final=np.multiply(a1,small_image)
    image[int(y):int(y)+height_required,int(x):int(x)+width_required]= image[int(y):int(y)+height_required,int(x):int(x)+width_required]+final
    counter+=1

"""saving our predicted Image"""


cv2.imwrite('/home/rgupta/Desktop/two_lens_result_new/stereo/pred1.exr',image)