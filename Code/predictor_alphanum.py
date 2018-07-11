# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 00:34:20 2018

@author: sohai
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:47:43 2018

@author: sohail
"""

from keras.models import load_model

classifier = load_model('model_2.h5')
values_map = dict()
with open('emnist-byclass-mapping.txt','r') as f:
    for lines in f.readlines():
        vals = list(map(int,lines.split()))
        values_map[vals[0]] = chr(vals[1])

count = 121
import numpy as np
from PIL import Image, ImageOps
img = Image.open('letter/letters_'+str(count)+'.png')
old_size = img.size  # old_size[0] is in (width, height) format
ratio = float(28)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
delta_w = 28 - new_size[0]
delta_h = 28 - new_size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
img = ImageOps.expand(img, padding)
img = img.convert('L')
img_tensor = img.resize((28,28))
img_tensor = np.array(img_tensor)
#img = np.rot90(img)
img_tensor_2 = img_tensor/ 255.
img_tensor_2 = np.expand_dims(img_tensor_2, axis=0)
img_tensor_2 = np.reshape(img_tensor_2, (len(img_tensor_2), 28, 28, 1))

pr = classifier.predict_classes(img_tensor_2)

prediction = values_map[pr[0]]
with open('output.txt','a') as f:
    f.write(prediction+" ")
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()