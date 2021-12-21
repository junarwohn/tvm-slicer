import time
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
import sys
try:
    partition_point = int(sys.argv[1])
except:
    partition_point = 72
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# Load Prerequisites
if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
    weights_url = "".join(
        [
            "https://github.com/fchollet/deep-learning-models/releases/",
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_old.h5"
else:
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_new.h5"



weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_resnet50 = keras.applications.resnet50.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_resnet50.load_weights(weights_path)


img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
# input preprocess
data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])

shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

target = 'cuda'

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
    front_lib, back_lib = relay.partition_graph(mod, [target, target], params=params, graph_config=lib.get_graph_json(), partition_point=partition_point)


local_lib_full_path = './model_build/local_lib_full_{}.tar'.format(str(partition_point))
local_lib_front_path = './model_build/local_lib_front_{}.tar'.format(str(partition_point))
local_lib_back_path = './model_build/local_lib_back_{}.tar'.format(str(partition_point))

lib.export_library(local_lib_full_path)
front_lib.export_library(local_lib_front_path)
back_lib.export_library(local_lib_back_path)
