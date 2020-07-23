#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:39:06 2020

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

if lens_type == 'two':
    disparity = config.get('TRAINING_TWO','dispar_path')
    left = config.get('TRAINING_TWO','left_path')
    right = config.get('TRAINING_TWO','right_path')
if lens_type == 'four':
    disparity = config.get('TRAINING_FOUR','dispar_path')
    middle = config.get('TRAINING_FOUR','middle_path')
    left = config.get('TRAINING_FOUR','left_path')
    right = config.get('TRAINING_FOUR','right_path')
    top = config.get('TRAINING_FOUR','top_path')
    bottom = config.get('TRAINING_FOUR','bottom_path')

if lens_type == 'six':
    disparity = config.get('TRAINING_SIX','dispar_path')
    middle = config.get('TRAINING_SIX','middle_path')
    left = config.get('TRAINING_SIX','left_path')
    right = config.get('TRAINING_SIX','right_path')
    top = config.get('TRAINING_SIX','top_path')
    top1 = config.get('TRAINING_SIX','top1_path')
    bottom = config.get('TRAININGSIX','bottom_path')
    bottom1 = config.get('TRAINING_SIX','bottom1_path')


if lens_type == 'twelve':
    disparity = config.get('TRAINING_TWELVE','dispar_path')
    middle = config.get('TRAINING_TWELVE','middle_path')
    left = config.get('TRAINING_TWELVE','left_path')
    right = config.get('TRAINING_TWELVE','right_path')
    top = config.get('TRAINING_TWELVE','top_path')
    top1 = config.get('TRAINING_TWELVE','top1_path')
    top2 = config.get('TRAINING_TWELVE','top2_path')
    top3 = config.get('TRAINING_TWELVE','top3_path')
    bottom = config.get('TRAINING_TWELVE','bottom_path')
    bottom1 = config.get('TRAINING_TWELVE','bottom1_path')
    bottom2 = config.get('TRAINING_TWELVE','bottom2_path')
    bottom3 = config.get('TRAINING_TWELVE','bottom3_path')
    bottom_left = config.get('TRAINING_TWELVE','bottom_left_path')
    top_right = config.get('TRAINING_TWELVE','top_right_path')

if lens_type == 'seven':
    disparity = config.get('TRAINING_SEVEN','dispar_path')
    middle = config.get('TRAINING_SEVEN','middle_path')
    first = config.get('TRAINING_SEVEN','first_path')
    second = config.get('TRAINING_SEVEN','second_path')
    third = config.get('TRAINING_SEVEN','third_path')
    fourth = config.get('TRAINING_SEVEN','fourth_path')
    fifth = config.get('TRAINING_SEVEN','fifth_path')
    sixth = config.get('TRAINING_SEVEN','sixth_path')


    
if lens_type == 'three':
    disparity = config.get('TRAINING_THREE','dispar_path')
    middle = config.get('TRAINING_THREE','middle')
    middle_left = config.get('TRAINING_THREE','middle_left')
    first = config.get('TRAINING_THREE','first')
    second = config.get('TRAINING_THREE','second')
    first_left = config.get('TRAINING_THREE','first_left')
    second_left = config.get('TRAINING_THREE','second_left')
    coor_values = config.get('TRAINING_THREE','coor_values')


""" we will get the list of all the files/Images used for training"""

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
    #print(img_path)
    dis_fullpath = os.path.join(dis_path,img_path)
    dis_list.append(dis_fullpath)
    print(dis_list)


"""To read all the ground truth Images we have made a function """
def disp(value1,value2,count,v_number):
    
    dis_img = cv2.imread(dis_list[count],cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    dis_height, dis_width, _ = dis_img.shape
    """ initializing variable for storing our values and to make our work easier,although in python we can directly save the values in 
      a variable we dont need to specify them""""
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
    
        if lens_type == 'four':
            if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                print(int(y),c,int(x),a)
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)    
            
        if lens_type == 'six':
            if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                print(int(y),c,int(x),a)
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)  
        if lens_type == 'twelve':
            if(70<=int(x)<width and a<=(width-70)) and (int(y)>=80 and c<=(height-80)):
                crop=dis_img[int(y):c,int(x):a]
                print('disparity')
                print(int(y),c,int(x),a)
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)  
        if lens_type == 'seven':
            if(0<=int(x)<width and a<= (width-210)) and (int(y) and c <= (height)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                print(int(y),c,int(x),a)
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop) 
        elif lens_type == 'three':
            if(70<=int(x)<width and a<= (width-70)) and (int(y) and c <= (height)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                print(int(y),c,int(x),a)
                cv2.imwrite(disparity+str(count)+'-'+str(v_number)+'.exr',crop)    
            
    return 1

"""initializing counter to store our rectangular training cutouts/data/lenses"""
counter1=1  
counter2=1

""" to get all the Images that we wanted to use for training  we are joining Image path with all the Images and reading all the
Images """
for img_path in dirs:
#    folder_name =os.path.splitext(img_path)[0]
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
    for r in range(0,6000):
        random_number = randint(0,6000)# taking 6000 lenses from each image, we can change this number also within the length of array lens 
        x,y = pixel[random_number]
        v_number+=1
        
        disp(x,y,count,v_number)#call disp function
    """ initializing variable for storing our values and to make our work easier,specify the path of path.properties file"""
        a =0
        c =0
    """ initialize counter to restrict the flow of program """
        i =0
        
        while i<1:
            i+=1
            c=int(y)+height_required
            a=int(x)+width_required
            
            """ cutting out rectangular boxes enclosing hexagonal Input data/lens for training for two-lens 
            and other Input lens type Input data and we are making it sure that all the neighbors exists for each lens
            that we are cutting out first which means when  we are cutting down one single lens and if we are using two-lens 
            input data it must have its immediate right lens and if we are using four lens and it must have all the immediate 
            three neighours and we are following same approach for every input data lens and we are making it
            sure by applying certain conditions that we must cut out all the lenses that we required by using os.path.isfile
            command and as we go along the code we will understand """
            
            if lens_type == 'two':
                if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                    crop=image[int(y):c, int(x):a]
                    print('Left')
                    cv2.imwrite(left+str(counter1)+'.png',crop)
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile(left+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    cv2.imwrite(right+str(counter1)+'.png',crop)
            
            if lens_type == 'four':
                if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
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
                    cv2.imwrite(top+str(counter1)+'.png',crop)
                if ( 35<=int(x)<width and a<= (width-35)) and ( c and (c+height_required)<=height)and os.path.isfile(top+str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x):a]
                    print('bottom')
                    print(c,c+height_required,int(x),a)
                    cv2.imwrite(bottom+str(counter1)+'.png',crop)
            
            if lens_type == 'six':

                if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
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
                    cv2.imwrite(top+str(counter1)+'.png',crop)
                    
             

                if (int(y) and int(y)-height_required>=0) and (0<=int(x)-width_required<=int(x)) and os.path.isfile(top+str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x)-width_required:int(x)]
                    print('top1')
                    print(int(y)-height_required,int(y),int(x)-width_required,int(x))
                    cv2.imwrite(top1+str(counter1)+'.png',crop)
                
                if ( 35<=int(x)<width and a<= (width-35)) and ( c and (c+height_required)<=height)and os.path.isfile(top1+str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x):a]
                    print('bottom')
                    print(c,c+height_required,int(x),a)
                    cv2.imwrite(bottom+str(counter1)+'.png',crop)
                
                if (c and (c+height_required)<=height)and (a and (a+width_required)<=width)and os.path.isfile(bottom+str(counter1)+'.png'):
                    crop=image[c:c+height_required,a:a+width_required]
                    print('bottom1')
                    print(c,c+height_required,a,a+width_required)
                    cv2.imwrite(bottom1+str(counter1)+'.png',crop)
                
            if lens_type == 'twelve_lens':
                if(70<=int(x)<width and a<=(width-70)) and (int(y)>=80 and c<=(height-80)):
                    crop=image[int(y):c,int(x):a]
                    print('middle_lens')
                    print(int(y),c,int(x),a)
                    cv2.imwrite(middle+str(counter1)+'.png',crop)
                
                if(int(y) and c<=height) and (a+width_required and (a+width_required)<=width) and os.path.isfile(middle+str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('right')
                    print(int(y),c,a+width_required,a+2*width_required)
                    cv2.imwrite(right+str(counter1)+'.png',crop)
                
                if(int(y) and c<=height) and (0<=int(x)-2*width_required<=int(x)) and os.path.isfile(right+str(counter1)+'.png'):
                    crop=image[int(y):c,int(x)-2*width_required:((int(x)-2*width_required)+width_required)]
                    print('left')
                    print(int(y),c,int(x)-2*width_required,((int(x)-2*width_required)+width_required))
                    cv2.imwrite(left+str(counter1)+'.png',crop)
                    
                if(70<=int(x)<width and a<=width-70) and ((int(y)-height_required and int(y)-2*width_required)>=0) and os.path.isfile(left+str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x):a]
                    print('top')
                    print(int(y)-2*height_required,int(y)-height_required,int(x),a)
                    cv2.imwrite(top+str(counter1)+'.png',crop)
            
                    
                if((int(y)-height_required and int(y)-2*height_required)>=0) and (0<=int(x)-width_required<=int(x)) and os.path.isfile(top+str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x)-width_required:int(x)]
                    print('top1')
                    print(int(y)-2*height_required,int(y)-height_required,int(x)-width_required,int(x))
                    cv2.imwrite(top1+str(counter1)+'.png',crop)
                    
                if((int(y)-height_required and int(y)-2*height_required)>=0) and (0<=int(x)-2*width_required<=int(x)) and os.path.isfile(top1+str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x)-2*width_required:(int(x)-2*width_required)+width_required]
                    print('top2')
                    print(int(y)-2*height_required,int(y)-height_required,int(x)-2*width_required,(int(x)-2*width_required)+width_required)
                    cv2.imwrite(top2+str(counter1)+'.png',crop)
                    
                if (0<=int(x)-2*width_required<=int(x)) and (int(y) and int(y)-height_required >=0) and os.path.isfile(top2+str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x)-2*width_required:(int(x)-2*width_required)+width_required]
                    print('top3')
                    print(int(y)-height_required,int(y),int(x)-2*width_required,(int(x)-2*width_required)+width_required)
                    cv2.imwrite(top3+str(counter1)+'.png',crop)
                            
                if(70<=int(x)<width and a<=width-70) and ((c+height_required and c+2*height_required)<=height) and os.path.isfile(top3+str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,int(x):a]
                    print('bottom')
                    print(c+height_required,c+2*height_required,int(x),a)
                    cv2.imwrite(bottom+str(counter1)+'.png',crop)
                    
                if((c+height_required and c+2*height_required)<=height)and (a and a+width_required <=width) and os.path.isfile(bottom+str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,a:a+width_required]
                    print('bottom1')
                    print(c+height_required,c+2*height_required,a,a+width_required)
                    cv2.imwrite(bottom1+str(counter1)+'.png',crop)
                
                if((c+height_required and c+2*height_required)<=height) and (a+width_required and a+2*width_required<=width) and os.path.isfile(bottom1+str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,a+width_required:a+2*width_required]
                    print('bottom2')
                    print(c+height_required,c+2*height_required,a+width_required,a+2*width_required)
                    cv2.imwrite(bottom2+str(counter1)+'.png',crop)
                    
                if (a+width_required and a+2*width_required<=width) and (c and c+height_required<=height) and os.path.isfile(bottom2+str(counter1)+'.png'):
                    crop=image[c:c+height_required,a+width_required:a+2*width_required]
                    print('bottom3')
                    print(c,c+height_required,a+width_required,a+2*width_required)
                    cv2.imwrite(bottom3+str(counter1)+'.png',crop)
    
                
              
                if (int(y) and int(y)-height_required >=0) and (a and a+width_required<=width) and os.path.isfile(bottom3+str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),a:a+width_required]
                    print('top right')
                    print(int(y)-height_required,int(y),a,a+width_required)
                    cv2.imwrite(top_right+str(counter1)+'.png',crop)
                
                if(c and c+height_required <=height) and (0<=int(x)-width_required<=int(x)) and os.path.isfile(top_right+str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x)-width_required:int(x)]
                    print('bottom left')
                    print(c,c+height_required,int(x)-width_required,int(x))
                    cv2.imwrite(bottom_left+str(counter1)+'.png',crop)
            
            if lens_type =='sevend':
                if (0<=int(x)<width and a<= (width-210)) and (int(y) and c <= (height)) :
                    crop=image[int(y):c, int(x):a]
                    print('middle')
                    print(int(y),c,int(x),a)
                    cv2.imwrite(middle+str(counter1)+'.png',crop)
                
                if (int(y) and c <= (height)) and (a and (a+width_required)<=width) and os.path.isfile(middle+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('first')
                    print(int(y),c,a,a+width_required)
                    counter1+=1
                    cv2.imwrite(first+str(counter1)+'.png',crop)
            
                if (int(y) and c <= (height)) and (a+width_required and (a+2*width_required)<=width) and os.path.isfile(first+str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('second')
                    print(int(y),c,a+width_required,a+2*width_required)
                    counter1+=1
                    cv2.imwrite(second+str(counter1)+'.png',crop)
            
                if (int(y) and c <= (height)) and (a+2*width_required and (a+3*width_required)<=width) and os.path.isfile(second+str(counter1)+'.png'):
                    crop=image[int(y):c,a+2*width_required:a+3*width_required]
                    print('third')
                    print(int(y),c,a+2*width_required,a+3*width_required)
                    counter1+=1
                    cv2.imwrite(three+str(counter1)+'.png',crop)
            
                if (int(y) and c <= (height)) and (a+3*width_required and (a+4*width_required)<=width) and os.path.isfile(three+str(counter1)+'.png'):
                    crop=image[int(y):c,a+3*width_required:a+4*width_required]
                    print('fourt')
                    print(int(y),c,a+3*width_required,a+4*width_required)
                    counter1+=1
                    cv2.imwrite(fourth+str(counter1)+'.png',crop)
                    
                if (int(y) and c <= (height)) and (a+4*width_required and (a+5*width_required)<=width) and os.path.isfile(fourth+str(counter1)+'.png'):
                    crop=image[int(y):c,a+4*width_required:a+5*width_required]
                    print('fifth')
                    print(int(y),c,a+4*width_required,a+5*width_required)
                    counter1+=1
                    cv2.imwrite(fifth+str(counter1)+'.png',crop)
            
                if (int(y) and c <= (height)) and (a+5*width_required and (a+6*width_required)<=width) and os.path.isfile(fifth+str(counter1)+'.png'):
                    crop=image[int(y):c,a+5*width_required:a+6*width_required]
                    print('sixth')
                    print(int(y),c,a+5*width_required,a+6*width_required)
                    counter1+=1
                    cv2.imwrite(sixth+str(counter1)+'.png',crop)
                   
            if lens_type == 'threed':
                if (70<=int(x)<width and a<= (width-70)) and (int(y) and c <= (height)) :
                    crop=image[int(y):c, int(x):a]
                    print('middle')
                    print(int(y),c,int(x),a)
                    cv2.imwrite(middle+str(counter1)+'.png',crop)
                    print('middle_l')
                    cv2.imwrite(middle_left+str(counter2)+'.png',crop)

                
               if (int(y) and c <= (height)) and (a and (a+width_required)<=width) and os.path.isfile(middle+str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('first')
                    print(int(y),c,a,a+width_required)
                    counter1+=1
                    cv2.imwrite(first+str(counter1)+'.png',crop)
                
               if (int(y) and c <= (height)) and (a+width_required and (a+2*width_required)<=width) and os.path.isfile(first+str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('second')
                    print(int(y),c,a+width_required,a+2*width_required)
                    counter1+=1
                    cv2.imwrite(second+str(counter1)+'.png',crop)
                    
                
               if ( int(y) and c <= height) and ( 0<=(int(x)-width_required)<=int(x))and os.path.isfile(middle_left+str(counter2)+'.png'):
                    crop=image[int(y):c,int(x)-width_required:int(x)]
                    print('left_first')
                    print(int(y),c,int(x)-width_required,int(x))
                    counter2+=1
                    cv2.imwrite(first_left+str(counter2)+'.png',crop)
                
               if ( int(y) and c <= height) and ( 0<=(int(x)-2*width_required)<=int(x))and os.path.isfile(first_left+str(counter2)+'.png'):
                    crop=image[int(y):c,int(x)-2*width_required:int(x)-width_required]
                    print('left_second')
                    print(int(y),c,int(x)-2*width_required,int(x)-width_required)
                    counter2+=1
                    cv2.imwrite(second_left+str(counter2)+'.png',crop)
                    counter2+=1
                
        
        counter1+=1

                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
