#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:06:12 2020

@author: rgupta
"""


""" Importing libraries """


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns
import natsort
from sklearn.metrics import mean_squared_error


""" path for reading ground hexagonal Image """
img1=cv2.imread(r'C:\Users\Home\Desktop\new_test_Image\org2new.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1.shape
print(img1)

""" path for reading opencv raw predcited Image """

img2=cv2.imread(r'C:\Users\Home\Desktop\new_test_Image\raw\raw.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1.shape
len(img2)

""" path for reading opencv cross  predcited Image """
img3=cv2.imread(r'C:\Users\Home\Desktop\new_test_Image\cross\cross.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


""" as we are just interested in the pixels with values, we will remove all the pixels with zero values and 
nan values """


""" finding the coordinates of pixels having value zero """

img3index = np.argwhere((img3==0))

""" converting zeros to nan """

for i in range(0,len(img3index)):
    a,b = img3index[i]
    img3[a][b] = np.nan

    
""" extracting number of pxels having having values """
x1 = np.count_nonzero(img1)
y = img2[~np.isnan(img2)]
y1 =len(y)
z = img3[~np.isnan(img3)]
z1 = len(z)

""" names to plot on x-axis """
images = ['resnet','raw','cross']
""" list of the pixel numbers from each image """
pixels = [x1,y1,z1]
print(pixels)


""" plotting th epixel comparison graph and also x-label. y-label and title name we can give any 
name as per our convenience """
x = np.arange(len(images))          
plt.bar(x,height= pixels,width=0.4)
#plt.bar(x,height=y1)
plt.xlabel('Predicted Images')
plt.ylabel('Number od Pixels')
plt.grid()
plt.title('Pixel comparision')
plt.xticks(x,images)
plt.xticks(rotation=60)

plt.ticklabel_format(axis='y',useOffset=False,style='plain')
plt.tight_layout()

""" path for saving the plotting graph """
plt.savefig(r'C:\Users\Home\Desktop\new_test_Image\length1.png')












