# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:39:48 2020

@author: Home
"""

""" Importing libraries """

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns
import natsort
from sklearn.metrics import mean_squared_error

"""For results from our deep learning method and stereo block matching method use our ground hexagonal image, specify path where it is saved """

img1=cv2.imread(r'C:\Users\Home\Desktop\compare_result\org1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1.shape

""" specify path of our predicted Image """

print(img1)
img2=cv2.imread(r'C:\Users\Home\Desktop\compare_result\stereo_result1\pred1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)



""" as we are calculating ,mean swuared error and bad pixel ratio so we just want to take the pixels which have some values 
that is why we are converting all the zero values into nan so that it wont contribute to our mean squared error and bad pixel 
ratio"""

img1index = np.argwhere((img1==0))



for i in range(0,len(img1index)):
    a,b = img1index[i]
    img2[a][b] = np.nan
    
    
for i in range(0,len(img1index)):
    a,b = img1index[i]
    img1[a][b] = np.nan



""" taking pixels with values and ignoring all the nan values"""
x = img1[~np.isnan(img1)]
y = img2[~np.isnan(img2)]

""" calculating mean squared error"""
mse = mean_squared_error(x,y)
mse
""" calculating bad pixel ratio """
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

""" path for storing mean squared values in a txt file and we can use the same file for every methid as 
we are appending the values and we can put the name of method we are using"""

f = open(r'C:\Users\Home\Desktop\compare_result\msee.txt', 'a')
print('Test loss:', mse)
print('bm', mse, file = f)
f.close()

""" path for storing values of bad pixel ratio in a txt file we can use the same file for every methid as 
we are appending the values and we can put the name of method we are using"""


f = open(r'C:\Users\Home\Desktop\compare_result\bpre.txt', 'a')
print('Test loss:', bpr)
print('bm', bpr, file = f)
f.close()