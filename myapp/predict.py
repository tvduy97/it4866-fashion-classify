from __future__ import absolute_import, division, print_function
import cv2
import numpy as np


def handle_uploaded_file(file_path, model):
    # img = cv2.imread(file_path, 0)
    image = cv2.imread(file_path)
    r = 150.0 / image.shape[1]
    dim = (150, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    lower_white = np.array([220, 220, 220], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    # could also use threshold
    mask = cv2.inRange(resized, lower_white, upper_white)
    res = cv2.bitwise_not(resized, resized, mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    img = gray
    img = img / 255.0
    cv2.imshow('image', img)
    cv2.waitKey(0)
    img = cv2.resize(img,  (28, 28))
    img.reshape((28, 28))
    img = np.expand_dims(img, axis=0)
    predictions_single = model.predict(img)
    print(predictions_single)
    result = np.argmax(predictions_single[0])
    return {'type': int(result)}
