#!/usr/bin/env python
from glob import glob
import cv2
import math
import numpy as np
import sys
from enhancer import FingerprintImageEnhancer

unified_dim = (128, 128)
ddepth = cv2.CV_16S
image_enhancer = FingerprintImageEnhancer()

for i in range(50):
    folder = './dataset/sub' + str(i+1) + '/*.jpg'
    jpgs = glob(folder)
    for j in jpgs:
        img = cv2.imread(j)
        old_size = img.shape[:2]

        if old_size[0] > old_size[1]:
            dim_difference = old_size[0] - old_size[1]
        elif old_size[1] > old_size[0]:
            dim_difference = old_size[1] - old_size[0]

        img = cv2.resize(img, unified_dim, interpolation=cv2.INTER_LANCZOS4)

        if(len(img.shape) > 2):                               # convert image into gray if necessary
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        out = image_enhancer.enhance(img)     # run image enhancer
        image_enhancer.save_enhanced_image(j[:-3] + 'png')
