# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta

import numpy as np
import pandas as pd

data = pd.read_csv("fer2013.csv")
image_height, image_width = 48, 48

X_train = np.zeros((28709, 1, 48, 48), dtype=int)
Y_train = np.zeros((28709, 7), dtype=int)
for i in range(0, 28709):
    X = data['pixels'][i]
    X_train[i] = np.fromstring(X, dtype=int, sep=" ").reshape(1, image_height, image_width)
    Y = data['emotion'][i]
    Y_train[i][Y] = 1
           
X_test = np.zeros((3589, 1, 48, 48), dtype=int)
Y_test = np.zeros((3589, 7), dtype=int)
counter = 0
for i in range(28710, 32299):
    X = data['pixels'][i]
    X_test[counter] = np.fromstring(X, dtype=int, sep=" ").reshape(1, image_height, image_width)
    Y = data['emotion'][i]
    Y_test[counter][Y] = 1
    counter += 1
             
##data augmentation:
#datagen = ImageDataGenerator(
#    featurewise_center=False,  # set input mean to 0 over the dataset
#    samplewise_center=False,  # set each sample mean to 0
#    featurewise_std_normalization=False,  # divide inputs by std of the dataset
#    samplewise_std_normalization=False,  # divide each input by its std
#    zca_whitening=False,  # apply ZCA whitening
#    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
#    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
#    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
#    horizontal_flip=True,  # randomly flip images
#    vertical_flip=False, data_format='channels_first')  # randomly flip images

#model architecture:
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                        input_shape=(1, X_train.shape[2], X_train.shape[3])))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.2))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.2))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# optimizer:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print('Training....')

#datagen.fit(X_train)
#
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128), 
#                        steps_per_epoch=len(X_train)/128, epochs=100, 
#                        callbacks=[ModelCheckpoint("fer_{epoch:02d}-{val_acc:.2f}.hdf5", monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)],
#                        validation_data=(X_test, Y_test))

model.fit(X_train, Y_train, epochs=200, batch_size=512,
          validation_split=0.3, shuffle=True, verbose=1,
          callbacks=[ModelCheckpoint("fer_{epoch:02d}-{acc:.2f}.hdf5", monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)])

# model result:
train_score = model.evaluate(X_train, Y_train, batch_size=256, verbose=1)
print('\nTraining Loss: ', train_score[0])
print('Training Accuracy: ', train_score[1])
test_score = model.evaluate(X_test, Y_test, batch_size=256, verbose=1)
print('\nTest Loss: ', test_score[0])
print('Test Accuracy: ', test_score[1])


#model.save("FER {}%".format(str(test_score[1])[2:4]))
print("Network trained and saved as FER")

