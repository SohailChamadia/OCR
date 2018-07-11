# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:01:25 2018

@author: sohai
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:12:44 2018

@author: sohail
"""

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

input_img = Input(shape=(28, 28, 1))
input_shape = (28, 28, 1)
nb_filters = 64 # number of convolutional filters to use
pool_size = (2, 2) # size of pooling area for max pooling
kernel_size = (3, 3) # convolution kernel size
nb_classes = 62

classifier = Sequential()
classifier.add(Conv2D(nb_filters,
                        kernel_size,
                        padding='valid',
                        input_shape=input_shape,
                        activation='relu'))
classifier.add(Conv2D(nb_filters,
                        kernel_size,
                        activation='relu'))

classifier.add(MaxPooling2D(pool_size=pool_size))
classifier.add(Dropout(0.25))
classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(nb_classes, activation='softmax'))

classifier.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

import numpy as np
from keras.utils import np_utils
import pandas as pd

train = pd.read_csv("emnist-byclass-train.csv").values
test = pd.read_csv("emnist-byclass-test.csv").values

print('train shape:', train.shape)
print('test shape:', test.shape)

img_rows = 28
img_cols = 28
x_train = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)
x_test = test[:, 1:].reshape(test.shape[0], img_rows, img_cols, 1)

y_train = train[:, 0]
y_test = test[:, 0] 

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)
datagen.fit(x_train)
classifier.fit_generator(datagen.flow(x_train, Y_train, batch_size=256),
                         steps_per_epoch=len(x_train) / 256, epochs=10,
                         verbose=1,
                         validation_data=(x_test, Y_test))
classifier.save('Identifier_alpha.h5')
