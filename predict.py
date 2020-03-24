import numpy as np
import cv2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

size = 64

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=340)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy')

model.load_weights('model.h5')

img = cv2.imread('temp.jpg', 0)
img = cv2.resize(img, (64, 64))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print(pred.max())