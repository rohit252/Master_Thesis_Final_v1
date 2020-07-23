#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:56:11 2020

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
from numpy import asarray
from numpy import save 
import configparser

""" argument for running the whole program fron console """



lens_type = sys.argv[1]
print(lens_type)


""" reading the path from path.properties file,specify the path of path.properties file"""

config = configparser.ConfigParser()
config.readfp(open('/home/rgupta/final_files_v1/preparing_dataset/path.properties'))

""" here we will get path from where we have to read images for generating data for prediction """

path = config.get('PREDICTION', 'path' )
print(path)

""" here we will get path where we have to save the prediction data that we are generating """


if lens_type == 'two':
    left = config.get('PREDICTION_TWO','left_path')
    right = config.get('PREDICTION_TWO','right_path')
    coordinate = config.get('PREDICTION_TWO','coor_values')
if lens_type == 'four':
    middle = config.get('PREDICTION_FOUR','middle_path')
    left = config.get('PREDICTION_FOUR','left_path')
    right = config.get('PREDICTION_FOUR','right_path')
    top = config.get('PREDICTION_FOUR','top_path')
    bottom = config.get('PREDICTION_FOUR','bottom_path')
    coor_values = config.get('PREDICTION_FOUR','coor_values')

if lens_type == 'six':
    middle = config.get('PREDICTION_SIX','middle_path')
    left = config.get('PREDICTION_SIX','left_path')
    right = config.get('PREDICTION_SIX','right_path')
    top = config.get('PREDICTION_SIX','top_path')
    top1 = config.get('PREDICTION_SIX','top1_path')
    bottom = config.get('PREDICTION_SIX','bottom_path')
    bottom1 = config.get('PREDICTION_SIX','bottom1_path')
    coor_values = config.get('PREDICTION_SIX','coor_values')

if lens_type == 'twelve':
    middle = config.get('PREDICTION_TWELVE','middle_path')
    left = config.get('PREDICTION_TWELVE','left_path')
    right = config.get('PREDICTION_TWELVE','right_path')
    top = config.get('PREDICTION_TWELVE','top_path')
    top1 = config.get('PREDICTION_TWELVE','top1_path')
    top2 = config.get('PREDICTION_TWELVE','top2_path')
    top3 = config.get('PREDICTION_TWELVE','top3_path')
    bottom = config.get('PREDICTION_TWELVE','bottom_path')
    bottom1 = config.get('PREDICTION_TWELVE','bottom1_path')
    bottom2 = config.get('PREDICTION_TWELVE','bottom2_path')
    bottom3 = config.get('PREDICTION_TWELVE','bottom3_path')
    bottom_left = config.get('PREDICTION_TWELVE','bottom_left_path')
    top_right = config.get('PREDICTION_TWELVE','top_right_path')
    coor_values = config.get('PREDICTION_TWELVE','coor_values')

    
if lens_type == 'sevend':
    middle = config.get('PREDICTION_SEVEND','middle_path')
    first = config.get('PREDICTION_SEVEND','first_path')
    second = config.get('PREDICTION_SEVEND','second_path')
    third = config.get('PREDICTION_SEVEND','third_path')
    fourth = config.get('PREDICTION_SEVEND','fourth_path')
    fifth = config.get('PREDICTION_SEVEND','fifth_path')
    sixth = config.get('PREDICTION_SEVEND','sixth_path')
    coor_values = config.get('PREDICTION_SEVEND','coor_values')

    
if lens_type == 'threed':
    middle = config.get('PREDICTION_THREED','middle')
    middle_left = config.get('PREDICTION_THREED','middle_left')
    first = config.get('PREDICTION_THREED','first')
    second = config.get('PREDICTION_THREED','second')
    first_left = config.get('PREDICTION_THREED','first_left')
    second_left = config.get('PREDICTION_THREED','second_left')
    coor_values = config.get('PREDICTION_THREED','coor_values')


""" we will get the list of all the files/Images used for prediction in our case we are just taking a single full Image
and for creating data that can be used for prediction"""

dirs = os.listdir(path)
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
""" list for storing the coordinates of the Input lens/data """
coor_values=[]


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
        if(0<=px_coor1<height) and (0<=px_coor2<width):
            pixel.append(px_coor)
            
 
"""initializing counter to store our rectangular training cutouts/data/lenses"""
           
counter1 =1 
counter2 = 1

  
 """ to get all the Images that we wanted to use for training  we are joining Image path with all the Images and reading all the
Images """ 
for img_path in dirs:
    print(img_path)
    folder_name =os.path.splitext(img_path)[0]
    fullpath = os.path.join(path,img_path)
    image = cv2.imread(fullpath)
    height, width, _ = image.shape
   
    """ here we are taking all the lenses of an Image for prediction as we want to to prediction on full Image"""
    
    for z in range(0,len(pixel)):
        x,y=pixel[z]
        
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
            """ here we are cutting down rectangular boxes enclosing our hexagonal lenses that we can used for prediction
            and like the training here also we are making it sure to cut down all the neighbours of a lens that are required
            """
            if lens_type =='two':
                if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                    crop=image[int(y):c, int(x):a]
                    print('Left')
                    print(left)
                    cv2.imwrite(left+str(counter1)+'.png',crop)
                    coor_values.append((int(x),int(y)))
                
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile(left+str(counter1)+'.png'):
                    
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    cv2.imwrite(right+str(counter1)+'.png',crop)
                    
            if lens_type == 'four':
                if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
                    crop=image[int(y):c, int(x):a]
                    print('middle_lens')
                    print(int(y),c,int(x),a)
                    cv2.imwrite( middle +str(counter1)+'.png',crop)
                    coor_values.append((int(x),int(y)))
                    

                    
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile( middle +str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    print(int(y),c,a,a+width_required)
                    cv2.imwrite( right +str(counter1)+'.png',crop)
                
                if ( int(y) and c <= height) and ( 0<=(int(x)-width_required)<=int(x))and os.path.isfile( right +str(counter1)+'.png'):
                    crop=image[int(y):c,int(x)-width_required:int(x)]
                    print('left')
                    print(int(y),c,int(x)-width_required,int(x))
                    cv2.imwrite( left +str(counter1)+'.png',crop)
                
                if ( 35<=int(x)<width and a<= (width-35)) and ( int(y) and (int(y)-height_required)>=0)and os.path.isfile( left +str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x):a]
                    print('top')
                    print(int(y)-height_required,int(y),int(x),a)
                    cv2.imwrite( top +str(counter1)+'.png',crop)

                
                if ( 35<=int(x)<width and a<= (width-35)) and ( c and (c+height_required)<=height)and os.path.isfile( top +str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x):a]
                    print('bottom')
                    print(c,c+height_required,int(x),a)
                    cv2.imwrite( bottom +str(counter1)+'.png',crop)

                
            if lens_type == 'six': 
                if (35<=int(x)<width and a<= (width-35)) and (int(y)>=40 and c <= (height-40)):
                    crop=image[int(y):c, int(x):a]
                    print('middle_lens')
                    print(int(y),c,int(x),a)
                    cv2.imwrite( middle +str(counter1)+'.png',crop)

                    coor_values.append((int(x),int(y)))
        
            
                if ( int(y) and c <= height) and ( a and (a+width_required)<=width) and os.path.isfile( middle +str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('right')
                    print(int(y),c,a,a+width_required)
                    cv2.imwrite( right +str(counter1)+'.png',crop)

            
                if ( int(y) and c <= height) and ( 0<=(int(x)-width_required)<=int(x))and os.path.isfile( right +str(counter1)+'.png'):
                    crop=image[int(y):c,int(x)-width_required:int(x)]
                    print('left')
                    print(int(y),c,int(x)-width_required,int(x))
                    cv2.imwrite( left + str(counter1)+'.png',crop)

                
                if ( 35<=int(x)<width and a<= (width-35)) and ( int(y) and (int(y)-height_required)>=0)and os.path.isfile( left +str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x):a]
                    print('top')
                    print(int(y)-height_required,int(y),int(x),a)
                    cv2.imwrite( top +str(counter1)+'.png',crop)

            
                if (int(y) and int(y)-height_required>=0) and (0<=int(x)-width_required<=int(x)) and os.path.isfile( top +str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x)-width_required:int(x)]
                    print('top1')
                    print(int(y)-height_required,int(y),int(x)-width_required,int(x))
                    cv2.imwrite( top1 +str(counter1)+'.png',crop)

            
                if ( 35<=int(x)<width and a<= (width-35)) and ( c and (c+height_required)<=height)and os.path.isfile( top1 +str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x):a]
                    print('bottom')
                    print(c,c+height_required,int(x),a)
                    cv2.imwrite( bottom +str(counter1)+'.png',crop)

            
                if (c and (c+height_required)<=height)and (a and (a+width_required)<=width)and os.path.isfile( bottom +str(counter1)+'.png'):
                    crop=image[c:c+height_required,a:a+width_required]
                    print('bottom1')
                    print(c,c+height_required,a,a+width_required)
                    cv2.imwrite(bottom1 +str(counter1)+'.png',crop)

            if lens_type == 'twelve':
                if(70<=int(x)<width and a<=(width-70)) and (int(y)>=80 and c<=(height-80)):
                    crop=image[int(y):c,int(x):a]
                    print('middle_lens')
                    print(int(y),c,int(x),a)
                    coor_values.append((int(x),int(y)))
                    cv2.imwrite( middle +str(counter1)+'.png',crop)

                    
                if(int(y) and c<=height) and (a+width_required and (a+width_required)<=width) and os.path.isfile( middle +str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('right')
                    print(int(y),c,a+width_required,a+2*width_required)
                    cv2.imwrite( right +str(counter1)+'.png',crop)

                      
                if(int(y) and c<=height) and (0<=int(x)-2*width_required<=int(x)) and os.path.isfile( right +str(counter1)+'.png'):
                    crop=image[int(y):c,int(x)-2*width_required:((int(x)-2*width_required)+width_required)]
                    print('left')
                    print(int(y),c,int(x)-2*width_required,((int(x)-2*width_required)+width_required))
                    cv2.imwrite( left +str(counter1)+'.png',crop)
                
                if(70<=int(x)<width and a<=width-70) and ((int(y)-height_required and int(y)-2*width_required)>=0) and os.path.isfile( left +str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x):a]
                    print('top')
                    print(int(y)-2*height_required,int(y)-height_required,int(x),a)
                    cv2.imwrite( top +str(counter1)+'.png',crop)
            
                
                if((int(y)-height_required and int(y)-2*height_required)>=0) and (0<=int(x)-width_required<=int(x)) and os.path.isfile( top +str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x)-width_required:int(x)]
                    print('top1')
                    print(int(y)-2*height_required,int(y)-2,int(x)-width_required,int(x))
                    cv2.imwrite( top1 +str(counter1)+'.png',crop)
                    
                if((int(y)-height_required and int(y)-2*height_required)>=0) and (0<=int(x)-2*width_required<=int(x)) and os.path.isfile(top1+str(counter1)+'.png'):
                    crop=image[int(y)-2*height_required:int(y)-height_required,int(x)-2*width_required:(int(x)-2*width_required)+width_required]
                    print('top2')
                    print(int(y)-2*height_required,int(y)-height_required,int(x)-2*width_required,(int(x)-2*width_required)+width_required)
                    cv2.imwrite( top2 +str(counter1)+'.png',crop)
                    
                if (0<=int(x)-2*width_required<=int(x)) and (int(y) and int(y)-height_required >=0) and os.path.isfile( top2 +str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),int(x)-2*width_required:(int(x)-2*width_required)+width_required]
                    print('top3')
                    print(int(y)-height_required,int(y),int(x)-2*width_required,(int(x)-2*width_required)+width_required)
                    cv2.imwrite( top3 +str(counter1)+'.png',crop)
                        
                if(70<=int(x)<width and a<=width-70) and ((c+height_required and c+2*height_required)<=height) and os.path.isfile( top3 +str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,int(x):a]
                    print('bottom')
                    print(c+height_required,c+2*height_required,int(x),a)
                    cv2.imwrite(bottom +str(counter1)+'.png',crop)

                    
                if((c+height_required and c+2*height_required)<=height)and (a and a+width_required <=width) and os.path.isfile( bottom +str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,a:a+width_required]
                    print('bottom1')
                    print(c+height_required,c+2*height_required,a,a+width_required)
                    cv2.imwrite( bottom1 +str(counter1)+'.png',crop)
            
                if((c+height_required and c+2*height_required)<=height) and (a+width_required and a+2*width_required<=width) and os.path.isfile( bottom1+str(counter1)+'.png'):
                    crop=image[c+height_required:c+2*height_required,a+width_required:a+2*width_required]
                    print('bottom2')
                    print(c+height_required,c+2*height_required,a+width_required,a+2*width_required)
                    cv2.imwrite( bottom2 +str(counter1)+'.png',crop)

                    
                if (a+width_required and a+2*width_required<=width) and (c and c+height_required<=height) and os.path.isfile( bottom2 +str(counter1)+'.png'):
                    crop=image[c:c+height_required,a+width_required:a+2*width_required]
                    print('bottom3')
                    print(c,c+height_required,a+width_required,a+2*width_required)
                    cv2.imwrite( bottom3 +str(counter1)+'.png',crop)


            
          
                if (int(y) and int(y)-height_required >=0) and (a and a+width_required<=width) and os.path.isfile( bottom3 +str(counter1)+'.png'):
                    crop=image[int(y)-height_required:int(y),a:a+width_required]
                    print('top right')
                    print(int(y)-height_required,int(y),a,a+width_required)
                    cv2.imwrite( top_right +str(counter1)+'.png',crop)

                
                if(c and c+height_required <=height) and (0<=int(x)-width_required<=int(x)) and os.path.isfile( top_right +str(counter1)+'.png'):
                    crop=image[c:c+height_required,int(x)-width_required:int(x)]
                    print('bottom left')
                    print(c,c+height_required,int(x)-width_required,int(x))
                    cv2.imwrite( bottom_left +str(counter1)+'.png',crop)

            if lens_type == 'sevend':
                if (0<=int(x)<width and a<= (width-210)) and (int(y) and c <= (height)) :
                    crop=image[int(y):c, int(x):a]
                    print('middle')
                    print(int(y),c,int(x),a)
                    coor_values.append((int(x),int(y)))
                    cv2.imwrite( middle+str(counter1)+'.png',crop)

                    
                if (int(y) and c <= (height)) and (a and (a+width_required)<=width) and os.path.isfile( middle +str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('first')
                    print(int(y),c,a,a+width_required)
                    counter1+=1
                    cv2.imwrite( first +str(counter1)+'.png',crop)

                
                if (int(y) and c <= (height)) and (a+width_required and (a+2*width_required)<=width) and os.path.isfile( first +str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('second')
                    print(int(y),c,a+width_required,a+2*width_required)
                    counter1+=1
                    cv2.imwrite( second +str(counter1)+'.png',crop)

                    
                if (int(y) and c <= (height)) and (a+2*width_required and (a+3*width_required)<=width) and os.path.isfile( second +str(counter1)+'.png'):
                    crop=image[int(y):c,a+2*width_required:a+3*width_required]
                    print('third')
                    print(int(y),c,a+2*width_required,a+3*width_required)
                    counter1+=1
                    cv2.imwrite( third +str(counter1)+'.png',crop)
                    
                if (int(y) and c <= (height)) and (a+3*width_required and (a+4*width_required)<=width) and os.path.isfile( third +str(counter1)+'.png'):
                    crop=image[int(y):c,a+3*width_required:a+4*width_required]
                    print('fourt')
                    print(int(y),c,a+3*width_required,a+4*width_required)
                    counter1+=1
#                    cv2.imwrite('//home//rgupta//Desktop//horizontal_images//prediction//horizontal//'+str(counter1)+'.png',crop)
                    cv2.imwrite( fourth+str(counter1)+'.png',crop)

                    
                if (int(y) and c <= (height)) and (a+4*width_required and (a+5*width_required)<=width) and os.path.isfile( fourth +str(counter1)+'.png'):
                    crop=image[int(y):c,a+4*width_required:a+5*width_required]
                    print('fifth')
                    print(int(y),c,a+4*width_required,a+5*width_required)
                    counter1+=1
                    cv2.imwrite( fifth +str(counter1)+'.png',crop)

                    
                if (int(y) and c <= (height)) and (a+5*width_required and (a+6*width_required)<=width) and os.path.isfile( fifth +str(counter1)+'.png'):
                    crop=image[int(y):c,a+5*width_required:a+6*width_required]
                    print('sixth')
                    print(int(y),c,a+5*width_required,a+6*width_required)
                    counter1+=1
                    cv2.imwrite( sixth +str(counter1)+'.png',crop)

            if lens_type =='threed':
                if (70<=int(x)<width and a<= (width-70)) and (int(y) and c <= (height)) :
                    crop=image[int(y):c, int(x):a]
                    print('middle')
                    print(int(y),c,int(x),a)
                    coor_values.append((int(x),int(y)))
                    cv2.imwrite( middle +str(counter1)+'.png',crop)
                    print('middle_l')
                    cv2.imwrite( middle_left +str(counter2)+'.png',crop)

                
                if (int(y) and c <= (height)) and (a and (a+width_required)<=width) and os.path.isfile( middle +str(counter1)+'.png'):
                    crop=image[int(y):c,a:a+width_required]
                    print('first')
                    print(int(y),c,a,a+width_required)
                    counter1+=1
                    cv2.imwrite( first +str(counter1)+'.png',crop)
                
                if (int(y) and c <= (height)) and (a+width_required and (a+2*width_required)<=width) and os.path.isfile(first +str(counter1)+'.png'):
                    crop=image[int(y):c,a+width_required:a+2*width_required]
                    print('second')
                    print(int(y),c,a+width_required,a+2*width_required)
                    counter1+=1
                    cv2.imwrite( second +str(counter1)+'.png',crop)
                
                
                if ( int(y) and c <= height) and ( 0<=(int(x)-width_required)<=int(x))and os.path.isfile( middle_left +str(counter2)+'.png'):
                    crop=image[int(y):c,int(x)-width_required:int(x)]
                    print('left_first')
                    print(int(y),c,int(x)-width_required,int(x))
                    counter2+=1
                    cv2.imwrite(first_left +str(counter2)+'.png',crop)
                
                if ( int(y) and c <= height) and ( 0<=(int(x)-2*width_required)<=int(x))and os.path.isfile( first_left +str(counter2)+'.png'):
                    crop=image[int(y):c,int(x)-2*width_required:int(x)-width_required]
                    print('left_second')
                    print(int(y),c,int(x)-2*width_required,int(x)-width_required)
                    counter2+=1
                    cv2.imwrite(second_left+str(counter2)+'.png',crop)
                    counter2+=1
              
        counter1+=1
        

""" storing the coordinates of all the Input lens/data that we want to predict and later we can use
these coordinates to generate the full predicted Image"""
coor_values = asarray(np.array(coor_values))
save(coordinate,coor_values)

              
    

  
        
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    