import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import preprocess_input

categories = np.load('categories.npy')

size = 64

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=340)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy')

model.load_weights('model.h5')

#img = cv2.imread('temp.jpg', 0)
img = Image.open('temp.jpg')
img = img_to_array(img)
img = preprocess_input(img).astype(np.float32)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print(categories[pred.argmax()])