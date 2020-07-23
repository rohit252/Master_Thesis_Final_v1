# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:29:57 2020

@author: Home

"""

import cv2 
print(cv2.__version__)
import os
import numpy as np
import matplotlib.pyplot as plt


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    """ converting image in BGR format"""
    color = tuple(reversed(rgb_color))
    image[:] = color

    return image

width1, height1 = 3500, 3500

red = (0.015*35,0,0)
image = create_blank(width1, height1, rgb_color=red)
#print(image)

""" path for extracting positional_images from the folder"""

path = '/home/rgupta/Desktop/positional_images'
dirs = os.listdir(path)
print(dirs)
count =1

for img_path in dirs:
    number = img_path[:-4]
    number = number[10:]
    print(number)


    fullpath = os.path.join(path,img_path)
    red_channel_image = cv2.imread(fullpath,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
   
    a = abs((red_channel_image)) + 0.015
    #print(a)
    disparity = cv2.divide(image,a)
    """storing the images in the folder"""
    cv2.imwrite('/home/rgupta/Desktop/disparity_images'+'//'+number+'.exr',disparity)
    count+=1
    

        
    
    
    
    
    
    
    
    
    
    
    