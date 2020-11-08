import math

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from glob import glob
import random
import os

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    cval=255,
    fill_mode='constant')

classes = 50

# remove old pictures
test = './previewTest/*.png'
train = './previewTrain/*.png'
train_files = glob(train)
for f in train_files:
    os.remove(f)
test_files = glob(test)
for f in test_files:
    os.remove(f)

# generate files to train the net
for k in range(classes):
    folder = './dataset/sub' + str(k + 1) + '/*.png'
    jpgs = glob(folder)
    class_id = 0
    for j in jpgs:
        name = str(k + 1) + 'n'
        print(name)
        img = load_img(j)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        save_dir = './previewTrain'
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_dir, save_prefix=name + str(class_id), save_format='png'):

            class_id += 1
            i += 1
            if i > 10:
                break

# isolate pictures to test the net
i = 0
for c in range(classes):
    i = i + 1
    source_dir = './previewTrain'
    dest_dir = './previewTest'
    folder = source_dir + '/' + str(i) + 'n*.png'
    jpgs = glob(folder)

    length = len(jpgs)
    # select % to test a net
    percent = math.ceil(length * 0.1)  # 10%
    if i == 1:
        print("length: " + str(length) + ", 10percent: " + str(percent))
    to_move = []
    for j in range(percent):
        rand = random.randint(1, length - 1)
        random_picture = jpgs[rand]
        while random_picture in to_move:
            rand = random.randint(1, length - 1)
            random_picture = jpgs[rand]
        to_move.append(random_picture)

    for tm in to_move:
        os.rename(str(tm), str(tm).replace(source_dir, dest_dir))
