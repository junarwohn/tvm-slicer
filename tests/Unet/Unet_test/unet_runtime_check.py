import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# from UNetKerasAS import UNet as UnetAS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

model_config = [0,0,2,0]
# model_config = [0,0,0,0]
model = keras.models.load_model("unet_as_{}_{}_{}_{}.h5".format(*model_config))
input_data = np.random.normal(0,1,(1, 256,256,3))

print(model_config)
for i in range(10):
    model.predict(input_data)

stime = time.time()
for i in range(100):
    model.predict(input_data)

print(time.time() - stime)

print(model.count_params())