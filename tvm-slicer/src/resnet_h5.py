#import tvm
#from tvm import te
#import tvm.relay as relay
#from tvm.contrib.download import download_testdata
#from tensorflow import keras
#import tensorflow as tf
#import numpy as np
#
#weights_url = "".join(
#        [
#            " https://storage.googleapis.com/tensorflow/keras-applications/",
#            "resnet/resnet152_weights_tf_dim_ordering_tf_kernels.h5",
#            ]
#        )
#weights_file = "resnet152_keras_new.h5"
#
#
#weights_path = download_testdata(weights_url, weights_file, module="keras")
#keras_resnet152 = keras.applications.resnet.ResNet152(
#    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
#)
#keras_resnet152.load_weights(weights_path)
#
#keras_resnet152.save('resnet152_224.h5') 

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tvm.contrib.download import download_testdata
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np

keras_resnet152 = tf.keras.models.load_model("./resnet152_224.h5")

img_path = "./data/frames/0.jpg"
img = cv2.resize(cv2.imread(img_path), (224,224))
print(img.shape)
cv2.imshow("aa", img)
cv2.waitKey(0)
# input preprocess
data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])
print("input_1", data.shape)

synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
# confirm correctness with keras output
keras_out = keras_resnet152.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print("Keras top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))

