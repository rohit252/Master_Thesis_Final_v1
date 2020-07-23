# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:53:20 2020

@author: Home
"""


""" Importing libraries"""
import numpy as np
import cv2
from PIL import Image
import glob
import os.path, sys
import os
from math import *
from random import*
import random
import natsort
from numpy import asarray
from numpy import save

""" width of an hexagonal lens"""
size = 35
"""width and height of our hexagonal Input lens/data"""
width_required=35
height_required=40
"""Initializing list for storing the values of coordinates of an Image"""
pixel = []
"""height and width of our full synthetic(RGB) and ground truth Image """
height = 3500
width = 3500
""" storing the coordinates that we have used of an Image that we want to predict """
coor_values=[]
counter=0

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
            
""" path for Extracting the Image that we have used for prediction """         
          
dis_path = '/home/rgupta/Desktop/testing_images/disparity'
dis_dirs = os.listdir(dis_path)
dis_dirs=natsort.natsorted(dis_dirs)
#print(dis_dirs)

"""cutting out the Input data from full Image that we can used later for our evaluation purpose and all these Input 
hexagonal lens/data are enclosed in a rectangular box and later we will using another script we will convert them
into hexagonal patch as our area of interest is hexagonal because w have used hexagonal array grid 
in our camera setup """

for img_path in dis_dirs:
    #print(img_path)
   
    dis_fullpath = os.path.join(dis_path,img_path)
#    print(dis_list)
    dis_img = cv2.imread(dis_fullpath,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#    dis_height, dis_width, _ = dis_img.shape
    for z in range(0,len(pixel)):
        x,y=pixel[z]
        
        a =0
        c =0
        i =0
        
        while i<1:
            i+=1
            c=int(y)+height_required
            a=int(x)+width_required
            if (int(x)<width and a<= (width-35)) and (int(y) and c <= (height)):
                crop=dis_img[int(y):c, int(x):a]
                print('disparity')
                """path for storing all the cutouts """
                cv2.imwrite('//home//rgupta//Desktop//original_disparity//disparity//'+str(counter)+'.exr',crop)  
                coor_values.append((int(x),int(y)))
                
                counter+=1
            
                
#len(coor_values)                
                
""" storing the coordinates of the cutouts in an numpy array """
coor_values = asarray(np.array(coor_values))
""" path for saving the array """
save('/home/rgupta/Desktop/original_disparity/coor_values.npy',coor_values)    


