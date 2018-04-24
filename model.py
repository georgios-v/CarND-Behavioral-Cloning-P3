#!/usr/bin/env python

import argparse
import csv
import cv2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys 
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D as crop2, Convolution2D as conv2
from keras.layers.pooling import MaxPooling2D as mpool2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def load_data(args):
    images = []
    measurements = []

    with open(os.path.join(args.data_dir, 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i, cor in enumerate([0.0, 0.2, -0.2]):
                img = cv2.imread(line[i])
                steering = float(line[3]) + cor
                images.append(img)
                measurements.append(steering)
    X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=args.test_size)
    args.img_shape = X_train[0].shape
    return X_train, X_valid, y_train, y_valid
    

def build_model(args):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=args.img_shape))
    model.add(crop2(cropping=((60, 25), (0, 0))))
    model.add(conv2(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(conv2(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(conv2(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(conv2(64, (3, 3), activation='relu'))
    model.add(conv2(64, (3, 3), activation='relu'))
    model.add(Dropout(dropout=args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate))
    return model


def train_generator(args, X, y, augment=True):
    i = 0
    while True:
        X_ = np.empty([args.batch_size, *args.img_shape])
        y_ = np.empty(args.batch_size)
        j = k = 0
        while True:
            img = X[i + j]
            steering = y[i + j]
            X_[j] = img
            y_[j] = steering
            k += 1
            if augment and k < args.batch_size - 1:
                X_[k] = cv2.flip(img, 1)
                y_[k] = steering * -1.0
                k += 1
            j += 1
            if k == args.batch_size:
                break
        yield X_, y_
        i += args.batch_size


def train(args, model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', monitor='val_loss', verbose=0, 
                                 save_best_only=args.save_best_only, mode='auto')
    history_object = model.fit_generator(train_generator(args, X_train, y_train, augment=True),
                                         steps_per_epoch=args.steps_per_epoch,
                                         epochs=args.epochs,
                                         verbose=1,
                                         callbacks=[checkpoint],
                                         validation_data=train_generator(args, X_valid, y_valid, augment=False),
                                         validation_steps=len(X_valid),
                                         workers=arg.n_procs - 1,
                                         use_multiprocessing=True,
                                         shuffle=True)
    return history_object


def plot_accuracy(history_object):
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(tf.Session(config=config))
    
    parser = argparse.ArgumentParser(description='CarND-Behavioral-Cloning-P3')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='../CarND-Behavioral-Cloning-data')
    parser.add_argument('-t', help='test size',             dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='dropout probability',   dest='keep_prob',         type=float, default=0.7)
    parser.add_argument('-e', help='number of epochs',      dest='epochs',            type=int,   default=10)
    parser.add_argument('-s', help='steps per epoch',       dest='steps_per_epoch',   type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.0011)
    parser.add_argument('-c', help='save best models only', dest='save_best_only',    action='store_true', default=False)
    parser.add_argument('-p', help='plot accuracy',         dest='plot',              action='store_true', default=False)
    args = parser.parse_args()
    arg.n_procs = multiprocessing.cpu_count()
    return args


if __name__ == '__main__':
    args = config()
    data = load_data(args)
    model = build_model(args)
    history_object = train(args, model, *data)
    if args.plot:
        plot_accuracy(history_object)
    
