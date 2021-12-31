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
from copy import deepcopy

import json
import pygraphviz as pgv

from simple_unet import UNet

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

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((512, 512))

model = UNet(in_dim=3, out_dim=1, num_filter=16)

# Preprocess (data type conversion to float32, NCHW order)
img_data = np.array(img)[np.newaxis, :].astype("float32")
img_data = preprocess_input(img_data).transpose([0, 3, 1, 2])
print("image shape :", img_data.shape)

# Get IRModule and Parameters
shape_dict = {"input_1": img_data.shape}
mod, params = relay.frontend.from_keras(model, shape_dict)

target = 'cuda'

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

local_dev = tvm.cuda()

# Save json graph
show_graph(lib['get_graph_json'](), file_name='unet_multi')