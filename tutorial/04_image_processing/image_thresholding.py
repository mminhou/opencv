""" #.03
  learn Simple thresholding, Adaptive thresholding, Otsu's thresholding etc.
  functions : cv2.threshold, cv2.adaptiveThreshold etc.

"""

# 여기 다시 공부해야함

import cv2
import numpy as np
from matplotlib import pyplot as plt

### Simple Thresholding
# img = cv2.imread('../factory/gradient.png', 0)
# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BiNARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()

### Adaptive Thresholding : 이거 하려면 image가 gray scale이어야 함
# _img = cv2.imread('../factory/sudokusmall.png')
# img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# # ADAPTIVE_THRESH_MEAN_C :  threshold value is the mean of neighbourhood area.
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# titles = ['Original Image', 'Global Thresholding (v =127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()

### Otsu's Binarization
img = cv2.imread('noisy2.png', 0)



#     4
#     5 img = cv2.imread('noisy2.png',0)
#     6
#     7 # global thresholding
#     8 ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#     9
#    10 # Otsu's thresholding
#    11 ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    12
#    13 # Otsu's thresholding after Gaussian filtering
#    14 blur = cv2.GaussianBlur(img,(5,5),0)
#    15 ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    16
#    17 # plot all the images and their histograms
#    18 images = [img, 0, th1,
#    19           img, 0, th2,
#    20           blur, 0, th3]
#    21 titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#    22           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#    23           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
#    24
#    25 for i in xrange(3):
#    26     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#    27     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#    28     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#    29     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#    30     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#    31     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#    32 plt.show()
#
#
#
# img = cv2.imread('noisy2.png',0)
#     2 blur = cv2.GaussianBlur(img,(5,5),0)
#     3
#     4 # find normalized_histogram, and its cumulative distribution function
#     5 hist = cv2.calcHist([blur],[0],None,[256],[0,256])
#     6 hist_norm = hist.ravel()/hist.max()
#     7 Q = hist_norm.cumsum()
#     8
#     9 bins = np.arange(256)
#    10
#    11 fn_min = np.inf
#    12 thresh = -1
#    13
#    14 for i in xrange(1,256):
#    15     p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
#    16     q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
#    17     b1,b2 = np.hsplit(bins,[i]) # weights
#    18
#    19     # finding means and variances
#    20     m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
#    21     v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
#    22
#    23     # calculates the minimization function
#    24     fn = v1*q1 + v2*q2
#    25     if fn < fn_min:
#    26         fn_min = fn
#    27         thresh = i
#    28
#    29 # find otsu's threshold value with OpenCV function
#    30 ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    31 print thresh,ret