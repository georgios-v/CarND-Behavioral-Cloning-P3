#!/usr/bin/env python

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D as conv2
from keras.layers.pooling import MaxPooling2D as mpool2

images = []
measurements = []

with open('../CarND-Behavioral-Cloning-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        images.append(cv2.imread(line[0]))
        measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

img_shape = X_train[0].shape

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=img_shape))
model.add(conv2(6, 5, 5, activation='relu'))
model.add(mpool2())
model.add(mpool2())
model.add(conv2(6, 5, 5, activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
