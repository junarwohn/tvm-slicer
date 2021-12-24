import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import numpy as np
from PIL import Image
import sys
import os

import json
import pygraphviz as pgv

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]), node['name'] + '[{}]'.format(node_idx))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')
    # return jupyterImage(A.draw(format='png', prog='dot'))

try:
    partition_point = int(sys.argv[1])
except:
    partition_point = 72

try:
    override_compile = bool(sys.argv[2])
except:
    override_compile = False

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# Load Keras Resnet50 Model
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

# Load (224, 224, 3) cat model.
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess (data type conversion to float32, NCHW order)
img_data = np.array(img)[np.newaxis, :].astype("float32")
img_data = preprocess_input(img_data).transpose([0, 3, 1, 2])

# Get IRModule and Parameters
shape_dict = {"input_1": img_data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

# Build model
target = 'cuda'

lib_path = "./model_build/"
model_name = "resnet50"

full_lib_name = "{}-libfull-{}.so".format(model_name, partition_point)
front_lib_name = "{}-libfront-{}.so".format(model_name, partition_point)
back_lib_name = "{}-libback-{}.so".format(model_name, partition_point)

lib_lists = os.listdir(lib_path)

# If lib file already exist, load library
if front_lib_name in lib_lists and back_lib_name in lib_lists and override_compile == False:
    print("Lib file alread exist. Load file...")
    full_lib = tvm.runtime.load_module(lib_path + full_lib_name)
    front_lib = tvm.runtime.load_module(lib_path + front_lib_name)
    back_lib = tvm.runtime.load_module(lib_path + back_lib_name)

# Build and partition_graph
else:
    print("Lib file doesn't exist. Build lib...")
    # partition_graph by addressing json graph and partition_point.
    with tvm.transform.PassContext(opt_level=3):
        full_lib = relay.build(mod, target, params=params)
        front_lib, back_lib = relay.partition_graph(mod, [target, target], params=params, graph_config=full_lib.get_graph_json(), partition_point=partition_point)
    full_lib.export_library(lib_path + full_lib_name)
    front_lib.export_library(lib_path + front_lib_name)
    back_lib.export_library(lib_path + back_lib_name)

# Set device and make graph executor front and back. 
local_dev = tvm.cuda()

full_model = debug_executor.create(full_lib["get_graph_json"](), full_lib, local_dev, dump_root="/tmp/tvmdbg")
# front_model = debug_executor.GraphModuleDebug(front_lib["default"](local_dev), device=local_dev, graph_json_str=front_lib["get_graph_json"](), dump_root="/tmp/tvmdbg")
# front_model_debug = debug_executor.create(front_lib["get_graph_json"](), front_lib, local_dev, dump_root="/tmp/tvmdbg")

# front_model = graph_executor.GraphModule(front_lib["default"](local_dev))
# back_model = graph_executor.GraphModule(back_lib["default"](local_dev))

time_start = time.time()

show_graph(full_lib["get_graph_json"](), "full_model")
# show_graph(front_lib["get_graph_json"](), "front_model")
# show_graph(back_lib["get_graph_json"](), "back_model")

front_out_shape = (1, 512, 28, 28)

for i in range(1):
    # Load Data
    # img = Image.open(img_path).resize((224, 224))
    # data = np.array(img)[np.newaxis, :].astype("float32")
    # data = preprocess_input(data).transpose([0, 3, 1, 2])
    input_data = tvm.nd.array(img_data)
    
    # Execute front debug model
    # front_model_debug.set_input('input_1', input_data)
    # front_model_debug.set_input(**params)
    # front_model_debug.run()
    # front_out_debug = front_model_debug.get_output(0, tvm.nd.empty(front_out_shape, dtype='float32')).numpy()

    # Execute Full Model
    full_model.set_input('input_1', input_data)
    full_model.set_input(**params)
    full_model.run()
    full_out = tvm.nd.empty((1,1000), dtype='float32')
    print(full_out)
    # full_model.get_output(0, full_out).numpy()
    full_model.debug_get_output("tvmgen_default_fused_nn_softmax", full_out)
    # full_model.exit()
"""
    # Execute front model
    front_model.set_input('input_1', input_data)
    # front_model.set_input(**params)
    front_model.run()
    # front_model.debug_get_output()
    front_out = front_model.get_output(0).numpy()
    input_data = tvm.nd.array(front_out)
    # input_data = tvm.nd.array(front_out_debug)

    print('input_shape', input_data.shape)

    # print(front_out_debug.shape, front_out.shape)
    # print(front_out_debug - front_out)
    # Execute back model
    back_model.set_input('input_1', input_data)
    back_model.run()
    back_out = back_model.get_output(0).numpy()
"""

print(full_out.shape)
total_time = time.time() - time_start
# top1_tvm = np.argmax(back_out)
top1_tvm = np.argmax(full_out)
print("total time : {}".format(total_time))

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
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))