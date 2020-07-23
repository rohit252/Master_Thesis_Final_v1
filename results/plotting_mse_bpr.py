# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:23:16 2020

@author: Home
"""

""" Importing libraries """
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import natsort

""" Initializing list """

li =[]
li1=[]
li2=[]
li11=[]


""" path for using bad pixel ratio file text values files """

for line in open('/home/rgupta/Desktop/all_results/two_lens_cross/mse.txt'):
    nums,fl, = line.split()
    li.append(nums)
    li1.append(fl)

""" path for using mean squared error text values files """

for line in open('/home/rgupta/Desktop/all_results/two_lens_cross/mse.txt'):
    nums,fl, = line.split()
    li.append(nums)
    li1.append(fl)

"""appending the values in the list created above"""
for i in range(0,len(li1)):
    li11.append(float(li1[i]))



""" plotting bar grapg for comaprison bwteen different methods """
x=np.arange(len(li))
plt.bar(x,height=li11,width=0.4)

plt.grid()
plt.title('Comparison of Models for Two-Lens with different variations')         
        
plt.xticks(x,li)
plt.xticks(rotation=60)
""" if calculating for mse then put MSE and if for bpr then put BPR 1.0"""
plt.ylabel('BPR 1.0')
#plt.ylabel('MSE')
""" dpeending on the type of input data we can put the name """
plt.xlabel('Two Lens Input Data')

plt.tight_layout()
"""path for storing the MSE graph """
plt.savefig("/home/rgupta/Desktop/all_results/two_lens_cross/comparision_mse.png")
"""path for storing the BPR graph """
plt.savefig("/home/rgupta/Desktop/all_results/two_lens_cross/comparision_bpr.png")

plt.close()


