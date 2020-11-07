from __future__ import print_function
from glob import glob
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2


train_pngs = glob('./previewTrain/*.png')
test_pngs = glob('./previewTest/*.png')
train_labels = []
test_labels = []
train_imgs = []
test_imgs = []

for j in train_pngs:
    train_imgs.append(cv2.imread(j))
    if j[16] == 'n':
        print(j[15])
        train_labels.append(int(j[15]))
    else:
        print(j[15:17])
        train_labels.append(int(j[15:17]))


for i in test_pngs:
    test_imgs.append(cv2.imread(i))
    if j[16] == 'n':
        print(j[15])
        test_labels.append(int(j[15]))
    else:
        print(j[15:17])
        test_labels.append(int(j[15:17]))


train_imgs = np.array(train_imgs, dtype=np.float32)
test_imgs = np.array(test_imgs, dtype=np.float32)

nRows, nCols, nDims = train_imgs.shape[1:]
input_shape = (nRows, nCols, nDims)

classes = np.unique(train_labels)
nClasses = len(classes)

train_imgs /= 255
test_imgs /= 255

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Get the InceptionV3 model so we can do transfer learning
base_inception = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(128, 128, 3))

# Add a global spatial average pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(512, activation='relu')(out)
out = Dense(512, activation='relu')(out)
predictions = Dense(nClasses, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False

# Compile
model.compile(Adam(lr=.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

batch_size = 90
epochs = 20

history = model.fit(train_imgs, train_labels_one_hot, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(test_imgs, test_labels_one_hot))

model.save_weights('test.h5')
model.evaluate(test_imgs, test_labels_one_hot)
Y_pred = model.predict(test_imgs)
y_pred = np.argmax(Y_pred, axis=1)
print(confusion_matrix(test_labels, y_pred))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
