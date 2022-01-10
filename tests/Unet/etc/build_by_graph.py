import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
# from tensorflow import keras
# from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import numpy as np
from PIL import Image
import sys
import os
import json

from simple_unet import UNet
# import copy

# img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
# img_path = download_testdata(img_url, "cat.png", module="data")
# img = Image.open(img_path).resize((512, 512))

model = UNet(in_dim=3, out_dim=1, num_filter=16)

# Preprocess (data type conversion to float32, NCHW order)
img = np.random.normal(0, 1, (3, 512, 512))
img_data = np.array(img)[np.newaxis, :].astype("float32")
print("image shape :", img_data.shape)

# Get IRModule and Parameters
shape_dict = {"input_1": img_data.shape}
mod, params = relay.frontend.from_keras(model, shape_dict)
target = 'cuda'

# Compile model
local_dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
model_1 = graph_executor.GraphModule(lib["default"](local_dev))
model_1.set_input('input_1', img_data)
model_1.run()
output_1 = model_1.get_output(0)

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target, params=params)
# model_1 = graph_executor.GraphModule(lib["default"](local_dev))
# model_1.set_input('input_1', img_data)
# model_1.run()

graph_json = json.loads(lib['get_graph_json']())
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_graph(mod, target, params=params, graph_config=json.dumps(graph_json))
# lib1, lib2 = relay.partition_graph(mod2, [target, target], params=params, graph_config=json.dumps(graph_json), partition_point=1)
# print("make done")
model_1 = graph_executor.GraphModule(lib["default"](local_dev))
model_1.set_input('input_1', img_data)
model_1.run()
output_1 = model_1.get_output(0)


"""
graph_json = json.loads(lib['get_graph_json']())

with open("graph_json1.json", "r") as json_file:
    json.load(graph_json, json_file)

relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json))

tvm_model1 = graph_executor.GraphModule(lib["default"](local_dev))

tvm_model1.set_input('input_1', img_data)
tvm_model1.run()

out1 = tvm_model1.get_output(0)

print(out1.shape)

with open("graph_json2.json", "r") as json_file:
    json.load(graph_json, json_file)

relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json))

tvm_model2 = graph_executor.GraphModule(lib["default"](local_dev))

tvm_model2.set_input('input_1', out1)
tvm_model2.run()

out2 = tvm_model2.get_output(0)

print(out2.shape)
"""