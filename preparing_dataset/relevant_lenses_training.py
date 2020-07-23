#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:34:49 2020

@author: rgupta
"""

"""Importing python and other Libraries  """



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
import configparser

""" argument for running the full program fron console """


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
    disparity = config.get('TRAINING_TWO_RELEVANT','dispar_path')
    left = config.get('TRAINING_TWO_RELEVANT','left_path')
    right = config.get('TRAINING_TWO_RELEVANT','right_path')
if lens_type =='four':
    disparity = config.get('TRAINING_FOUR_RELEVANT','dispar_path')
    left = config.get('TRAINING_FOUR_RELEVANT','left_path')
    right = config.get('TRAINING_FOUR_RELEVANT','right_path')
    top = config.get('TRAINING_FOUR_RELEVANT','top_path')
    bottom = config.get('TRAINING_FOUR_RELEVANT','bottom_path')


""" we will get the list of all the files/images used for training"""


dirs = os.listdir(path)

""" to sort all the files in a list we have used natsort command """ 
dirs=natsort.natsorted(dirs)
print(dirs)

"""Initializing the counter for saving our rectangular cutouts """

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
        hexagonal structure from these rectangular boxes, This is our ground truth data that we are generating
        and as we intersted in cutting down more relevnat lenses so we are taking all the ground truth lens whose variance 
        is more than and equal to one"""
        if lens_type == 'two':
            if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                crop=dis_img[int(y):c, int(x):a]
                var = np.var(crop)
                if var>=1:
                    print('disparity')
                    cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)    
    
        if lens_type == 'four':
            if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
                crop=dis_img[int(y):c, int(x):a]
                var = np.var(crop)
                if var>=1:
                    print('disparity')
                    print(int(y),c,int(x),a)
                    cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)  
            
        
    return 1

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
        random_number = randint(0,7000)# taking 7000 lenses from each image, we can change this number also within the length of array lens 
        x,y = pixel[random_number]
        v_number+=1
        disp(x,y,count,v_number)#call disp function
        
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
        """ cutting out rectangular boxes enclosing hexagonal Input data/lens for training for two-lens 
            and other Input lens type Input datas and we are making it sure that all the neighbors exists for each lens
            that we are cutting out first which means when  we are cutting down one single lens and if we are using two-lens 
            input data it must have its immediate right lens and if we are using four lens and it must have all the immediate 
            three neighours and  we are making it sure that for every ground truth lens we cut training lens by applying certain conditions that we must cut out all the lenses that we required by using os.path.isfile
            command """
            if lens_type == 'two':
                if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)) and os.path.isfile(disparity+str(count)+'-'+str(v_number)+'.exr'):
                    crop=image[int(y):c, int(x):a]
                    print('Left')
                    cv2.imwrite(left+str(counter1)+'.png',crop)
                
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile(left+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    cv2.imwrite(right+str(counter1)+'.png',crop)
            
            if lens_type == 'four':
                if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40))and os.path.isfile(disparity+str(count)+'-'+str(v_number)+'.exr'):
                    crop=image[int(y):c, int(x):a]
                    print('middle_lens')
                    print(int(y),c,int(x),a)
                    cv2.imwrite(middle+str(counter1)+'.png',crop)
                
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile(middle+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    print(int(y),c,a,a+width_required)
                    cv2.imwrite(right+str(counter1)+'.png',crop)
                       
                if ( int(y) and c <= height) and ( 0<=(int(x)-width_required)<=int(x))and os.path.isfile(right+str(counter1)+'.png'):
                    crop=image[int(y):c,int(x)-width_required:int(x)]
                    print('left')
                    print(int(y),c,int(x)-width_required,int(x))
                    cv2.imwrite(left+str(counter1)+'.png',crop)
                
                if ( 35<=int(x)<width and a<= (width-35)) and ( int(y) and (int(y)-height_required)>=0)and os.path.isfile(left+str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x):a]
                    print('top')
                    print(int(y)-height_required,int(y),int(x),a)
                    cv2.imwrite('//home//rgupta//Desktop//train_images2//top//'+str(counter1)+'.png',crop)
                    
                if ( 35<=int(x)<width and a<= (width-35)) and ( c and (c+height_required)<=height)and os.path.isfile(top+str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x):a]
                    print('bottom')
                    print(c,c+height_required,int(x),a)
                    cv2.imwrite(bottom+str(counter1)+'.png',crop)
                
            
        counter1+=1

                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
