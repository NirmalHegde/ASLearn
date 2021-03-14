##  imports
# import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import confusion_matrix
import itertools
import os

#  import variables from organizing.py
from organizing import train_batches, valid_batches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning) %matplotlib inline

#  Deep Learning Sequential
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=26, activation='softmax')
])

# #  Calculating Accuracy of Model
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x= train_batches,
#         steps_per_epoch=len(train_batches),
#          validation_data= valid_batches,
#          validation_steps= len(valid_batches),
#          epochs=10,
#          verbose=2
#          )
