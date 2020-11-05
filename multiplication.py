from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from glob import glob                                                           
import cv2
import numpy as np
jpgs = glob('./dataset/sub1/*.jpg')

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


for j in jpgs:
    name = 'test'
    img = load_img(j)  
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='previewTrain', save_prefix=name+str(i), save_format='jpg'):
        i += 1
        if i > 100:
            break  