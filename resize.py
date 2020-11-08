#!/usr/bin/env python
from glob import glob
import cv2
unified_dim = (128, 128)
ddepth = cv2.CV_16S

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

        cv2.imwrite(j[:-3] + 'png', img)
