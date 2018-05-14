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
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D as crop2, Convolution2D as conv2
from keras.layers.pooling import MaxPooling2D as mpool2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_data(args):
    images = []
    measurements = []

    with open(os.path.join(args.data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i, cor in enumerate([0.0, 0.2, -0.2]):
                img = line[i].strip()
                steering = float(line[3]) + cor
                images.append(img)
                measurements.append(steering)
    X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=args.test_size)
    args.img_shape = load_image(X_train[0]).shape
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
    model.add(Dropout(rate=args.keep_prob))
    model.add(Flatten())
    model.add(Dense(1152))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    return model


def train_generator(args, X, y, augment=True):
    max_index = len(X) - 1
    #print("max index:", max_index)
    i = 0
    reset = False
    while True:
        X_ = []  # np.empty([args.batch_size, *args.img_shape])
        y_ = []  # np.empty(args.batch_size)
        j = k = 0
        while True:
            if i + j > max_index:
                reset = True
                break
            img = load_image(X[i + j])
            #print(img)
            steering = y[i + j]
            X_.append(img)
            y_.append(steering)
            k += 1
            if augment and len(y_) < args.batch_size:
                X_.append((cv2.flip(img, 1)))
                y_.append(steering * -1.0)
                k += 1
            j += 1
            #print('i', i, 'j', j, 'k', k, 'y_', len(y_))
            if len(y_) == args.batch_size:
                break
            if i + j > max_index:
                reset = True
                break
        X_ = np.array(X_)
        y_ = np.array(y_)
        yield X_, y_
        if reset and augment:
             X, y = shuffle(X, y)
        if reset:
            i = 0
            reset = False
        else:
            i = i + j
        # print(i)


def train(args, model, X_train, X_valid, y_train, y_valid):
    print("train sz:", len(X_train))
    print("valid sz:", len(X_valid))
    checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', monitor='val_loss', verbose=0, 
                                 save_best_only=args.save_best_only, mode='auto')
    history_object = model.fit_generator(train_generator(args, X_train, y_train, augment=True),
                                         steps_per_epoch=2*len(X_train)//args.batch_size, # args.steps_per_epoch,
                                         epochs=args.epochs,
                                         verbose=1,
                                         callbacks=[checkpoint],
                                         validation_data=train_generator(args, X_valid, y_valid, augment=False),
                                         validation_steps=len(X_valid)//args.batch_size,
                                         workers=args.n_procs - 1,
                                         use_multiprocessing=False,
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
    parser = argparse.ArgumentParser(description='CarND-Behavioral-Cloning-P3')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='../CarND-Behavioral-Cloning-data')
    parser.add_argument('-t', help='test size',             dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='dropout probability',   dest='keep_prob',         type=float, default=0.7)
    parser.add_argument('-e', help='number of epochs',      dest='epochs',            type=int,   default=10)
    parser.add_argument('-s', help='steps per epoch',       dest='steps_per_epoch',   type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.001)
    parser.add_argument('-c', help='save best models only', dest='save_best_only',    action='store_true', default=False)
    parser.add_argument('-p', help='plot accuracy',         dest='plot',              action='store_true', default=False)
    parser.add_argument('--dry', help='print args and exit',dest='dry', action='store_true', default=False)
    args = parser.parse_args()
    args.n_procs = multiprocessing.cpu_count()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(tf.Session(config=config))
    
    return args


if __name__ == '__main__':
    args = config()
    print(args)
    data = load_data(args)
    if args.dry:
        print(len(data[0]))
        sys.exit(0)
    model = build_model(args)
    history_object = train(args, model, *data)
    if args.plot:
        plot_accuracy(history_object)
    
