from __future__ import absolute_import, division, print_function
import cv2
import numpy as np


def handle_uploaded_file(file_path, model):
    img = cv2.imread(file_path, 0)
    img = img / 255.0
    print(img)
    img = cv2.resize(img,  (28, 28))
    img.reshape((28, 28))
    img = np.expand_dims(img, axis=0)
    predictions_single = model.predict(img)
    print(predictions_single)
    result = np.argmax(predictions_single[0])
    return {'type': int(result)}
