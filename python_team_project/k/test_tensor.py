from keras.models import load_model
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np




model = load_model('./model/5540131915_5647912227')

img_compare = cv2.imread('./0.png')
img_compare = cv2.resize(img_compare, (64, 64))
img_array = image.img_to_array(img_compare)
img_batch = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_batch)
print(prediction[0])