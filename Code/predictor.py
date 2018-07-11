# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:47:43 2018

@author: sohail
"""

from keras.models import load_model

autoencoder = load_model('Generator.h5')
classifier = load_model('Identifier.h5')

import numpy as np
from PIL import Image
import PIL.ImageOps 

img = Image.open('n1.png')
img = img.convert('L')
img = PIL.ImageOps.invert(img)
img_tensor = img.resize((28,28))
img_tensor = np.array(img_tensor)
img_tensor = img_tensor/ 255.
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = np.reshape(img_tensor, (len(img_tensor), 28, 28, 1))

decoded_imgs = autoencoder.predict(img_tensor)
pr = classifier.predict_classes(img_tensor)

import matplotlib.pyplot as plt
plt.imshow(decoded_imgs.reshape(28, 28))
plt.show()