# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:47:38 2020

@author: Home
"""

"""Importing Libraries"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

""" For blcok-matching algorithm(raw and cross) results use our original ground truth Image specify path where the image is saved """

image1=cv2.imread('/home/rgupta/Desktop/first_final_images_remaining/disparity/1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image1=image1[:,:,2]

""" For results from our deep learning method and stereo block matching algorithm
use our ground hexagonal image, specify path where it is saved """

image1=cv2.imread('/home/rgupta/Desktop/original_images_pred/org1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

""" specify path of our predicted Image """

image2=cv2.imread('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/pred1.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


""" if we are using crosschecking method from blockmatching first convert the zero values into nan values """

img2index = np.argwhere((image2==0))

for i in range(0,len(img2index)):
    a,b = img2index[i]
    image2[a][b] = np.nan

""" checking difference between the each predicted pixel and ground truth pixel """
diff_image=cv2.absdiff(image1,image2)

""" path for saving the difference image """
cv2.imwrite('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/diff1.exr',diff_image)

""" converting our difference Image into color map with colorbar to see the difference values and 
specify path for saving the result """

norm_image = cv2.normalize(diff_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
norm_image = norm_image.astype(np.uint8)

plt.set_cmap('nipy_spectral')
plt.imshow(diff_image)
plt.colorbar()
plt.savefig('/home/rgupta/Desktop/all_results/two_lens_cross/resnet_cross/imgplot1.png',dpi=500)
plt.close()







