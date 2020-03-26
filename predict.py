import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

categories = np.load('categories.npy')

"""
size = 64

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=340)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy')

model.load_weights('model.h5')
"""

imheight, imwidth = 32, 32
num_classes = 340

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(680, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights('model_weights.h5')


#img = cv2.imread('temp.jpg', 0)
img = Image.open('image.jpg')
img = img_to_array(img)/255
print(img.shape)

img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print(categories[pred.argmax()])
