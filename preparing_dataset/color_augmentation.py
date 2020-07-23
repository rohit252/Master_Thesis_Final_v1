#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:50:53 2020

@author: rgupta
"""

"""Importing python Libraries  """



import numpy as np
import cv2
from PIL import Image
import glob
import os.path, sys
import os
from math import *
import pickle
from random import*
import random
import natsort
from albumentations import RGBShift
import matplotlib.pyplot as plt
import configparser

""" argument for running the whole program fron console """


lens_type = sys.argv[1]
print(lens_type)

""" reading the path from path.properties file,specify the path of path.properties file"""


config = configparser.ConfigParser()
config.readfp(open('/home/rgupta/final_files_v1/preparing_dataset/path.properties')) 

""" here we will get path from where we have to read images for generating training data """


path = config.get('TRAINING', 'training_path' )
dis_path = config.get('TRAINING', 'disparity_path' )

""" here we will get path where we have to save the training data that we are generating """


if lens_type =='two':
    disparity = config.get('TRAINING_TWO_COLOR','dispar_path')
    left = config.get('TRAINING_TWO_COLOR','left_path')
    right = config.get('TRAINING_TWO_COLOR','right_path')



""" we will get the list of all the files/Images used for prediction in our case we are just taking a single full Image
and for creating data that can be used for prediction"""

dirs = os.listdir(path)
dirs=natsort.natsorted(dirs)
print(dirs)
count = -1

""" width of the hexagonal lens """
size = 35

""" height and width of our hexagonal Input data/lens"""

width_required=35
height_required=40
""" list for storing all the pixel coordinates """

pixel = []
"""height and width of our full synthetic(RGB) and ground truth Image """

height = 3500
width = 3500

""" For storing ground truth data/lens we initialize list"""

dis_list=[]

""" generating the values of possible coordinates of an Image """ 

vx = size*(np.array([[1],
          [0]]))
vy = size*(np.array([[0.5],
                     [0.5*sqrt(3)]]))
for x_coor in range(-100,101):
    for y_coor in range(-100,101):
        px_coor = (x_coor*vx)+(y_coor*vy)+np.array([[1750],
                                                       [1750]])
        px_coor1,px_coor2 = px_coor
        px_coor = (px_coor1-17.5,px_coor2-20)
        px_coor1,px_coor2 = px_coor
        if(0<px_coor1<height) and (0<px_coor2<width):
            pixel.append(px_coor)
            

""" to get the list of all the grund truth files/Images"""

dis_dirs = os.listdir(dis_path)
""" sorting all the files """

dis_dirs=natsort.natsorted(dis_dirs)
print(dis_dirs)
"""In this for loop we are joining the path where our ground truth Images ae saved with all the Images"""

for img_path in dis_dirs:

    dis_fullpath = os.path.join(dis_path,img_path)
    dis_list.append(dis_fullpath)
    print(dis_list)

"""To read all the ground truth Images we have made a function """
    
def disp(value1,value2,count,v_number):
    dis_img = cv2.imread(dis_list[count],cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    dis_height, dis_width, _ = dis_img.shape
   """ initializing variable for storing our values and to make our work easier,although in python we can directly save the values in 
      a variable we dont need to specify them"""
    a =0
    c =0
   """ initialize counter to restrict the flow of program """

    i =0    
    while i<1:
        i+=1
        c=int(y)+height_required
        a=int(x)+width_required
        """ here we are cutting out all the hexagonal lenses enclosed in a rectangular boxes, basicllay we are cutting
        down rectangular boxes and later for our evaluation and prediction we will cutout original 
        hexagonal structure from these rectangular boxes,This is our ground truth data that we are generating"""
        if lens_type == 'two':
            
            if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)    
    return 1

""" defining function for color augmentation """
def augment_and_show(aug, image):
    image1 = aug(image=image)['image']

    return image1

"""initializing counter to store our rectangular training cutouts/data/lenses"""

counter1=1  

""" to get all the Images that we wanted to use for training  we are joining Image path with all the Images and reading all the
Images """

for img_path in dirs:
    print(img_path)
    folder_name =os.path.splitext(img_path)[0]

        
    fullpath = os.path.join(path,img_path)
    image = cv2.imread(fullpath)
    height, width, _ = image.shape
   """ counter for storing the ground data """

    count+=1
    v_number=0
    
    """ this is very important section of our program as here we are cutting down all the rectangular boxes 
    enclosing our hexagonal lenses for training and for each rectangular box we are also cutting down its corresponding
    ground truth data that is why we have made disp function so that we can call it here and to make sure that we
    are cuting down all the data correctly and from each full Image that we have used for training we can radomly take 
    some number of lenses for training """
               
    for r in range(0,7000):
        random_number = randint(0,7000) # taking 7000 lenses from each image, we can change this number also within the length of array lens 
        x,y = pixel[random_number]
        v_number+=1
        """ calling disp function """
        disp(x,y,count,v_number)
""" initializing variable for storing our values and to make our work easier,although in python we can directly save the values in 
      a variable we dont need to specify them"""

        a =0
        c =0
   """ initialize counter to restrict the flow of program """
        i =0
        
        while i<1:
            i+=1
            c=int(y)+height_required
            a=int(x)+width_required
       """ cutting out rectangular boxes enclosing hexagonal Input data/lens for training for two-lens input data and we are 
       making it sure that if we cutting down one lens we will also cut down its immediate neighbour and here we call 
       our color auhmentation function which we made above to augment the color of training lenses
       """"
            if lens_type=='two':
                    
                if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                    crop=image[int(y):c, int(x):a]
                    aug = RGBShift(r_shift_limit=1.5, g_shift_limit=0.5, b_shift_limit=-1.5, always_apply=True, p=0.5)
                    image_color = augment_and_show(aug, crop)
                    print('Left')
                    cv2.imwrite(left+str(counter1)+'.png',image_color)
                    
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile(left+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    aug = RGBShift(r_shift_limit=1.5, g_shift_limit=0.5, b_shift_limit=-1.5, always_apply=True, p=0.5)
                    image_color = augment_and_show(aug, crop)
                    print('right')
                    cv2.imwrite(right+str(counter1)+'.png',image_color)
                        
        counter1+=1

                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
