# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:49:08 2020

@author: Home
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns
import natsort
from sklearn.metrics import mean_squared_error



""" For blcok-matching algorithm results(raw and cross) use our original ground truth Image specify path where the image is saved """


img1=cv2.imread('/home/rgupta/Desktop/first_final_images_remaining/disparity/1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1=img1[:,:,2] #extracting just red channel values
print(img1)

"""specify path of our predicted Image"""
img2=cv2.imread('/home/rgupta/Desktop/all_results/two_lens_cross/resnet/pred1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)



""" if using cross-checking method of block matching then first converting zero into nan if its not converted, 
if the values are already nan then ignore these steps """


img2index = np.argwhere((img2==0))


for i in range(0,len(img2index)):
    a,b = img2index[i]
    img2[a][b] = np.nan
    
  
""" taking out the coordinate where we have nan values and converting exact coordinates in ground truth image to nan
for claculating the mean squated error and bad pixel ratio"""    
    
img2index = np.argwhere((np.isnan(img2)))



for i in range(0,len(img2index)):
    a,b = img2index[i]
    img1[a][b] = img2[a][b]
    
""" extracting only pixels with values"""

x = img1[~np.isnan(img1)]
y = img2[~np.isnan(img2)]


""" calculating mean squared error """

mse = mean_squared_error(x,y)
mse

""" calcuating bad pixel ratio """
error = abs(np.subtract(x,y))
bad_pixel=[]
for i in range(0,len(error)):
        a = error[i]
        if a >1.0:
            bad_pixel.append(a)
        
value = len(error)
bad_pixel_ratio  = np.divide(len(bad_pixel),value)

bpr = bad_pixel_ratio

bpr

""" path for storing mean squared values in a txt file we can use the same file for every methid as 
we are appending the values and we can put the name of method we are using"""



f = open("/home/rgupta/Desktop/all_results/two_lens_cross/mse.txt", 'a')
print('Test loss:', mse)
print('raw', mse,file = f)
f.close()


""" path for storing values of bad pixel ratio in a txt file we can use the same file for every methid as 
we are appending the values and we can put the name of method we are using"""


f = open("/home/rgupta/Desktop/all_results/two_lens_cross/bpr.txt", 'a')
print('Test loss:',bpr)
print('raw', bpr,file = f)
f.close()



