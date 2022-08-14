import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import time
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)).astype(np.float32) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

batch_size = 4
img_size = 512
in_dim = 3
out_dim = 1
num_filters = 16
epochs=100

model_path = "./model_{}th_512.h5"
model = keras.models.load_model(model_path.format(1))

cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
preprocess = make_preprocess('unet', 512)

while (cap.isOpened()):
    ret, frame = cap.read()
    frame = preprocess(frame)
    result =  model.predict(np.expand_dims(frame, axis=0))[0]
    th = cv2.resize(cv2.threshold(result, 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    frame[th == 1] = [0, 0, 255]
    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
