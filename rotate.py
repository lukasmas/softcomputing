#!/usr/bin/env python
from glob import glob                                                           
import cv2
import numpy as np
jpgs = glob('./*.jpg')
unified_dim = (128,128)
ddepth = cv2.CV_16S

for j in jpgs:
    img = cv2.imread(j)
    old_size = img.shape[:2]
    
    if old_size[0] > old_size[1]:
        dim_difference = old_size[0] - old_size[1]
        img = cv2.copyMakeBorder(img,0,0,0,dim_difference,cv2.BORDER_CONSTANT,value=(0,0,0))
    elif old_size[1] > old_size[0]:
        dim_difference = old_size[1] - old_size[0]
        img = cv2.copyMakeBorder(img,0,dim_difference,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        
    img = cv2.resize(img, unified_dim, interpolation = cv2.INTER_LANCZOS4)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    img1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    img3 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img4 = cv2.flip(img,-1)
    
    cv2.imwrite(j[:-3] + 'png', img)
    cv2.imwrite(j[:-4] + '10.' + 'png', img1)
    cv2.imwrite(j[:-4] + '20.' + 'png', img2)
    cv2.imwrite(j[:-4] + '30.' + 'png', img3)
    cv2.imwrite(j[:-4] + '40.' + 'png', img4)



    
    