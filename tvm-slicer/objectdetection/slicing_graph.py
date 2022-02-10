from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import os
import json
import pygraphviz as pgv
from argparse import ArgumentParser
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

parser = ArgumentParser()
parser.add_argument('--start_point', type=int, default=0)
parser.add_argument('--end_point', type=int, default=239)
parser.add_argument('--partition_point', type=int, default=60, help='set partition point')
parser.add_argument('--img_size', type=int, default=netw, help='set image size')
parser.add_argument('--model', type=str, default='yolov3', help='name of model')
parser.add_argument('--target', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', type=int, default=3, help='set opt_level')
args = parser.parse_args()

graph_json_raw = lib['get_graph_json']()

tvm_slicer = TVMSlicer(graph_json_raw)
#parser.add_argument('--start_point', type=int, default=0)
#parser.add_argument('--end_point', type=int, default=-1)
#parser.add_argument('--partition_point', type=int, default=0, help='set partition point')

graph_json_front_info = tvm_slicer.slice_graph(0, 60)
graph_json_back_info = tvm_slicer.slice_graph(60, 239)

#graph_json_back_info = TVMSlicer(graph_json_raw, [[0,10],[10,121]]).get_graph()
#graph_json_front_info, graph_json_back_info = TVMSlicer(graph_json_raw, [[0,9],[9,111]]).get_graph()

graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

# TODO adding final_shape 
# do 'extra' job to 

with open("./graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, netw, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_front['extra'] = {}
    graph_json_front['extra']['inputs'] = input_front
    graph_json_front['extra']['outputs'] = output_front
    json_file.write(json.dumps(graph_json_front))


with open("./graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, netw, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_back['extra'] = {}
    graph_json_back['extra']['inputs'] = output_front
    graph_json_back['extra']['outputs'] = output_back
    json_file.write(json.dumps(graph_json_back))
