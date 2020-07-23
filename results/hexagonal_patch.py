# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 00:56:59 2020

@author: Home
"""

import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import load
from numpy import asarray
from numpy import save


"""Creating a small BGR image of 35 by 40 with values ones"""
def create_Image(width, height, rgb_color=(0,0,0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float)
#     a = [np.zeros((3500,3500,3)).astype(object), np.zeros((3500,3500,3)).astype(object), np.zeros((3500,3500,3)).astype(object)]
    """ converting image in BGR format"""
    color = tuple(reversed(rgb_color))
#    image[:] = color
    image[:] = color

    return image

width1, height1 = 35, 40

all = (1,1,1)
image = create_Image(width1, height1, rgb_color=all)

""" As we just want to see the hexagonal structure in our Input lens/data,we inside this verices we keep all the
 ones values and outside the vertices we put zeros in order to clearly see the hexagonal structure of our Input data"""
vertices=[(20,0),(0,10),(0,0)],[(20,0),(35,0),(35,10)],[(0,40),(20,40),(0,30)],[(20,40),(35,40),(35,30)]
for x in range(len(vertices)):
    y=np.array(vertices[x])
    pts = y.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color=(0, 0, 0))

#plt.imshow(image)
a2=np.array(image,dtype=float)
a2.shape
small_image=asarray(a2)
""" saving our Image in numpy array form """
save('/home/rgupta/Desktop/small_image.npy',small_image)