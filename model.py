#!/usr/bin/env python

import argparse
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D as crop2, Convolution2D as conv2
from keras.layers.pooling import MaxPooling2D as mpool2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

epochs = 2
if len(sys.argv) > 1:
    epochs = int(sys.argv[1])

images = []
measurements = []

with open('../CarND-Behavioral-Cloning-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        for i, cor in enumerate([0.0, 0.2, -0.2]):
            img = cv2.imread(line[i])
            steering = float(line[3]) + cor
            images.append(img)
            measurements.append(steering)
            images.append(cv2.flip(img, 1))
            measurements.append(steering*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

img_shape = X_train[0].shape

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=img_shape))
model.add(crop2(cropping=((60, 25), (0, 0))))
model.add(conv2(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(conv2(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(conv2(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(conv2(64, (3, 3), activation='relu'))
model.add(conv2(64, (3, 3), activation='relu'))
model.add(Dropout(dropout=keep_prob))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, verbose=1)

model.save('model.h5')


print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

