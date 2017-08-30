# -*-coding:utf8-*-

import cv2
import numpy as np

img_source = './img_source/'
img_dist = './img_dist/'
img_name = '2012soranokiseki01l.jpg'
# img_name = 'ice cream11.jpg'
image = cv2.imread(img_source+img_name)
cv2.imshow('original',image)
# cv2.waitKey()

# 低通滤波
kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
rect = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'rect.jpg',rect)

# 高斯滤波
kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
gaussian = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'gaussian.jpg',gaussian)

# 锐化
kernel = np.array([[0,-2,0],[-2,9,-2],[0,-2,0]])
sharpen = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'sharpen.jpg',sharpen)

# 边缘检测
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edges = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'edges.jpg',edges)

# 浮雕
kernel = np.array([[-2,-2,-2,-2,0],[-2,-2,-2,0,2],[-2,-2,0,2,2],[-2,0,2,2,2],[0,2,2,2,2]])
emboss = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'emboss_step1.jpg',emboss)
emboss = cv2.cvtColor(emboss,cv2.COLOR_BGR2GRAY)
cv2.imwrite(img_dist+'emboss.jpg',emboss)

# 随便写一个
kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])/36
my_image = cv2.filter2D(image,-1,kernel)
cv2.imwrite(img_dist+'my_image.jpg',my_image)
