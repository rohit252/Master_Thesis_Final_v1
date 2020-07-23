# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:04:08 2020

@author: Home
"""


  
"""Importing libraries """

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns
import natsort
import math as m


""" For results from our deep learning method and stereo block matching algorithm
use our ground hexagonal image, specify path where it is saved """

img1=cv2.imread('/home/rgupta/Desktop/original_images_pred/org1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

""" For blcok-matching algorithm(raw and cross) results use our original ground truth Image specify path where the image is saved """


img1=cv2.imread('/home/rgupta/Desktop/first_final_images_remaining/disparity/1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1=img1[:,:,2] # extracting just red channel




"""specify path of our predicted Image"""

img2=cv2.imread('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/pred1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


""" if we are using crosschecking method from blockmatching first convert the zero values into nan values """

##### converting zero to nan
img2index = np.argwhere((img2==0))

for i in range(0,len(img2index)):
    a,b = img2index[i]
    img2[a][b] = np.nan
"""  taking the absolute difference between both the original and our predicted image"""
error = abs(np.subtract(img1,img2))
""" extracting the minimum value other than zero for plotting purpose """  
minval= np.min(img1[np.nonzero(img1)])
""" function for x_axis labelling """
def tic(a,b):
    list1=[]
    v=0
    for i in range(m.floor(minval),m.floor(img1.max())):
        v = i
        if i <= minval:
            list1.append(minval)
        if i > minval:
            list1.append(i)
    if v != img1.max():
        list1.append(img1.max())
    
    return list1


""" plotting scatter plot between error and ground truth(disparity values) """

plt.scatter(img1,error,s=0.0001)
plt.xlabel('Disparity')
"""calling function here"""
ticks = tic(img1.min(),img1.max())
ticks = list(set(ticks))
ticks = natsort.natsorted(ticks)
"""arranging ticks and limits"""
plt.xlim(minval,img1.max())
plt.xticks(ticks,rotation=60)
plt.ylabel('Error')
plt.ylim(ymin=0)
plt.title('Evaluation of "Disparity vs Error" ')
plt.tight_layout()
""" path for saving the plotting figure """
plt.savefig('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/evaluation.png')
plt.close()

            
""" plotting histogram between pixels and error """
error_min = np.nanmin((error))
error_max = np.nanmax((error))
x=np.arange(error_min,error_max,1)
x
plt.hist(error,
         bins=x,
         histtype='bar',
         edgecolor='black',
#         facecolor='yellow',
         rwidth=1.0,
         alpha=0.5)
plt.xlabel('Error')
plt.xticks(x)
plt.xlim(error_min,error_max)
plt.ylabel('Pixels')
plt.title("Evaluation of Errors in Different range ")
""" path for saving the plotting figure """
plt.savefig('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/evaluation_hist.png')
plt.close()




