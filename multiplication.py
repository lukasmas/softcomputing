from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from glob import glob
import random
import os
import cv2
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    cval=255,
    fill_mode='constant')

classes = 50

for k in range(classes):
    folder = './dataset/sub' + str(k + 1) + '/*.png'
    jpgs = glob(folder)
    id = 0
    for j in jpgs:
        name = str(k + 1) + 'n'
        print(name)
        img = load_img(j)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        #Trzeba dopracować :D
        # rand = random.random() * 10
        # if rand >= 9.0:
            save_dir = 'previewTrain'
        # else:
        #     save_dir = 'previewTest'

        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_dir, save_prefix=name + str(id), save_format='png'):

            id += 1
            i += 1
            if i > 5:
                break