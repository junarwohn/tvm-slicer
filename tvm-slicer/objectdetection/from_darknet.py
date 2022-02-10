# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile YOLO-V2 and YOLO-V3 in DarkNet Models
=============================================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This article is an introductory tutorial to deploy darknet models with TVM.
All the required models and libraries will be downloaded from the internet by the script.
This script runs the YOLO-V2 and YOLO-V3 Model with the bounding boxes
Darknet parsing have dependancy with CFFI and CV2 library
Please install CFFI and CV2 before executing this script

.. code-block:: bash

  pip install cffi
  pip install opencv-python
"""

# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys

# tvm, relay
import tvm
from tvm import te
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

import cv2

######################################################################
# Choose the model
# -----------------------
# Models are: 'yolov2', 'yolov3' or 'yolov3-tiny'

# Model name
MODEL_NAME = "yolov3"

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.

cfg_path = "../src/model/darknet/yolov3.cfg"
weights_path = "../src/model/darknet/yolov3.weights"
lib_path = "../src/model/darknet/libdarknet2.0.so"

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {"data": data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

def convert(o):
    if isinstance(o, numpy.int64): return int(o)  
    raise TypeError
    
######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
target = "cuda"
dev = tvm.cuda()

shape = {"data": data.shape}
print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

[neth, netw] = shape["data"][2:]  # Current image shape is 608x608
######################################################################
# Load a test image
# -----------------
test_image = "kite.jpg"
img_path = "../src/data/" + test_image

data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))

# set inputs
m.set_input("data", tvm.nd.array(data.astype(dtype)))
# execute
print("Running the test image...")

# detection
# thresholds
thresh = 0.5
nms_thresh = 0.45

m.run()
# get outputs
tvm_out = []
if MODEL_NAME == "yolov2":
    layer_out = {}
    layer_out["type"] = "Region"
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).numpy()
    layer_out["biases"] = m.get_output(1).numpy()
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
    layer_out["output"] = m.get_output(0).numpy().reshape(out_shape)
    layer_out["classes"] = layer_attr[4]
    layer_out["coords"] = layer_attr[5]
    layer_out["background"] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == "yolov3":
    for i in range(3):
        layer_out = {}
        layer_out["type"] = "Yolo"
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i * 4 + 3).numpy()
        layer_out["biases"] = m.get_output(i * 4 + 2).numpy()
        layer_out["mask"] = m.get_output(i * 4 + 1).numpy()
        out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
        layer_out["output"] = m.get_output(i * 4).numpy().reshape(out_shape)
        layer_out["classes"] = layer_attr[4]
        tvm_out.append(layer_out)

elif MODEL_NAME == "yolov3-tiny":
    for i in range(2):
        layer_out = {}
        layer_out["type"] = "Yolo"
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i * 4 + 3).numpy()
        layer_out["biases"] = m.get_output(i * 4 + 2).numpy()
        layer_out["mask"] = m.get_output(i * 4 + 1).numpy()
        out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
        layer_out["output"] = m.get_output(i * 4).numpy().reshape(out_shape)
        layer_out["classes"] = layer_attr[4]
        tvm_out.append(layer_out)
        thresh = 0.560

# do the detection and bring up the bounding boxes
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
    (netw, neth), (im_w, im_h), thresh, 1, tvm_out
)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = "coco.names"
coco_url = "../src/" + "data/" + coco_name
font_name = "arial.ttf"
font_url = "../src/" + "data/" + font_name
coco_path = download_testdata(coco_url, coco_name, module="data")
font_path = download_testdata(font_url, font_name, module="data")

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

tvm.relay.testing.yolo_detection.show_detections(img, dets, thresh, names, last_layer.classes)
tvm.relay.testing.yolo_detection.draw_detections(
    font_path, img, dets, thresh, names, last_layer.classes
)
img = img.transpose(1, 2, 0)
cv2.imshow("img-{}".format(test_image.split(".")[0]), img)
cv2.waitKey(0)
print(img.shape)
cv2.imwrite("./result-{}.jpg".format(test_image.split(".")[0]), 255 * img)


# import tensorflow as tf
# import tvm
# import tvm.relay as relay
# from tvm.contrib import graph_executor
# import numpy as np
# import os
import json
import pygraphviz as pgv

# os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

def shape_size(shape_list):
    result = 1
    for i in shape_list:
        result *= i
    return result

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(shape_size(json_data['attrs']['shape'][1][src[0]])), node['name'] + '[{}]'.format(node_idx) + '{}'.format(shape_size(json_data['attrs']['shape'][1][node_idx])))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

show_graph(lib['get_graph_json'](), "darknet_{}_lv_{}".format(target, 3))

with open("darknet_{}_lv_{}".format(target, 3), "w") as json_file:
    json_file.write(lib['get_graph_json']())